"""A thin, resilient wrapper around a local Ollama server.

Design goals:
- **Never let a node crash because the model server is down.** Every call
  reports success/failure via :class:`LLMResult`; callers fall back to
  deterministic behavior when ``ok`` is False.
- **Validated structured output.** :meth:`chat_structured` returns a parsed,
  schema-validated Pydantic object or ``None`` — no raw-JSON handling in nodes.
- **Cheap availability probing,** cached so a missing server doesn't add latency
  to every node in a run.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, TypeVar

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
    error: Optional[str] = None
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
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or settings.llm
        self._client = None
        self._available: Optional[bool] = None  # tri-state cache

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

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResult:
        if not self.available():
            return LLMResult(text="", ok=False, error="llm_unavailable", fallback=True)

        model = model or self.cfg.chat_model
        options: dict[str, object] = {
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": 0.9,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict

        last_err: Optional[str] = None
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
        model: Optional[str] = None,
    ) -> Optional[T]:
        """Return a validated ``schema`` instance, or ``None`` on any failure."""
        res = self.chat(system, user, model=model, json_mode=True, temperature=0.1)
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

    def embed(self, text: str, *, model: Optional[str] = None) -> Optional[list[float]]:
        """Return an embedding vector, or ``None`` if embeddings are unavailable."""
        if not self.available() or not text.strip():
            return None
        model = model or self.cfg.embed_model
        try:
            resp = self.client.embeddings(model=model, prompt=text)
            emb = resp["embedding"] if isinstance(resp, dict) else getattr(resp, "embedding", None)
            return [float(x) for x in emb] if emb else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("embeddings failed (%s); semantic score will use lexical fallback", exc)
            return None


_default_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Process-wide singleton so availability is probed once per run."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
