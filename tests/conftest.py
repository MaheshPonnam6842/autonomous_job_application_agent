"""Test configuration.

Force the fully-offline deterministic path so the suite never depends on a
running Ollama server. This must run before any ``v_final.config`` import, which
pytest guarantees by loading conftest first.
"""

import os

os.environ.setdefault("JOBAGENT_LLM_ENABLED", "0")
os.environ.setdefault("JOBAGENT_ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "_artifacts"))
