"""LLM access layer for the v_final agent.

Nodes never import ``ollama`` directly. They go through :class:`OllamaClient`,
which centralizes host config, retries, JSON-mode structured output, embeddings,
and — crucially — graceful degradation when the model server is unavailable.
"""

from v_final.llm.client import LLMResult, OllamaClient, get_client
from v_final.llm.schemas import JDExtraction

__all__ = ["OllamaClient", "LLMResult", "get_client", "JDExtraction"]
