"""A thin, resilient wrapper around a local Ollama server.

Design goals:
- **Never let a node crash because the model server is down.** Every call
  reports success/failure via :class:`LLMResult`; callers fall back to
  deterministic behavior when ``ok`` is False.
- **Validated structured output.** :meth:`chat_structured` returns a parsed,
  schema-validated Pydantic object or ``None`` — no raw-JSON handling in nodes.
- **Fast on local CPU.** Embeddings are memoized (the JD is embedded once per
  run, not on every loop rescore), the model is kept warm between calls via
  ``keep_alive``, and ``num_thread`` / ``num_ctx`` are tunable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from v_final.config import LLMConfig, settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    """Outcome of a chat call. ``fallback=True`` means the server was skipped
    (disabled or unreachable) rather than failing mid-request."""

    text: str
    ok: bool
    error: str | None = None
    fallback: bool = False


def _extract_json_block(text: str) -> str:
    """Pull the first JSON object out of a model response, tolerating code fences."""
    if not text:
        return ""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


class OllamaClient:
    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or settings.llm
        self._client = None
        self._available: bool | None = None  # tri-state cache
        self._embed_cache: dict[str, list[float]] = {}  # memoized embeddings
        self._embed_ok: bool | None = None  # None=unknown, False=model missing (skip)

    @property
    def client(self):  # lazy import so the package loads without ollama installed
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.cfg.host, timeout=self.cfg.request_timeout)
        return self._client

    def available(self, force: bool = False) -> bool:
        """True if the model server answers. Cached after the first probe."""
        if not self.cfg.enabled:
            return False
        if self._available is not None and not force:
            return self._available
        try:
            self.client.list()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - any failure means "treat as down"
            logger.warning("Ollama unavailable at %s (%s); using deterministic fallbacks",
                           self.cfg.host, exc)
            self._available = False
        return self._available

    @staticmethod
    def _content(resp: object) -> str:
        """Read message content across ollama-python response shapes (dict / object)."""
        try:
            return resp["message"]["content"]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            msg = getattr(resp, "message", None)
            return getattr(msg, "content", "") if msg is not None else ""

    def _options(self, temperature: float | None, num_predict: int | None) -> dict[str, object]:
        opts: dict[str, object] = {
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": 0.9,
            "num_ctx": self.cfg.num_ctx,
        }
        if self.cfg.num_thread > 0:
            opts["num_thread"] = self.cfg.num_thread
        if num_predict is not None:
            opts["num_predict"] = num_predict
        return opts

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        if not self.available():
            return LLMResult(text="", ok=False, error="llm_unavailable", fallback=True)

        model = model or self.cfg.chat_model
        options = self._options(temperature, num_predict)

        last_err: str | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options=options,
                    format="json" if json_mode else "",
                    keep_alive=self.cfg.keep_alive,
                )
                return LLMResult(text=self._content(resp).strip(), ok=True)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                logger.warning("chat attempt %d/%d failed: %s",
                               attempt + 1, self.cfg.max_retries + 1, exc)
        return LLMResult(text="", ok=False, error=last_err)

    def chat_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        model: str | None = None,
        num_predict: int | None = None,
    ) -> T | None:
        """Return a validated ``schema`` instance, or ``None`` on any failure."""
        res = self.chat(system, user, model=model, json_mode=True,
                        temperature=0.1, num_predict=num_predict)
        if not res.ok:
            return None
        raw = _extract_json_block(res.text)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("structured output was not valid JSON: %s", exc)
            return None
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            logger.warning("structured output failed schema validation: %s", exc)
            return None

    def embed(self, text: str, *, model: str | None = None) -> list[float] | None:
        """Return an embedding vector, or ``None`` if embeddings are unavailable.

        Memoized by (model, content hash): the same text — e.g. the JD across
        every loop rescore — is embedded exactly once per process.
        """
        if not self.available() or not text.strip():
            return None
        if self._embed_ok is False:   # embed model already proven missing — skip the round-trip
            return None
        model = model or self.cfg.embed_model
        key = f"{model}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        try:
            try:
                resp = self.client.embeddings(model=model, prompt=text, keep_alive=self.cfg.keep_alive)
            except TypeError:  # older ollama without keep_alive kwarg
                resp = self.client.embeddings(model=model, prompt=text)
            emb = resp["embedding"] if isinstance(resp, dict) else getattr(resp, "embedding", None)
            vec = [float(x) for x in emb] if emb else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("embeddings failed (%s); semantic score will use lexical fallback", exc)
            self._embed_ok = False   # don't retry a missing embed model this run
            return None
        if vec is not None:
            self._embed_ok = True
            self._embed_cache[key] = vec
        return vec


_default_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    """Process-wide singleton so availability is probed once and the embedding
    cache is shared across every node in a run."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
