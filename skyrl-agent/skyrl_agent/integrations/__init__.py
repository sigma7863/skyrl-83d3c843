# Always register the lightweight OpenAI-compatible backend.
from . import openai  # noqa: F401

# The training backends (skyrl_train, verl, tinker) pull in heavy, mutually
# exclusive dependencies. Import them lazily so inference-only / minimal setups
# that only use the ``openai_server`` backend don't require the full training
# stack. Each registers itself on import via ``register_backend``.
for _mod in (".skyrl_train", ".verl", ".tinker"):
    try:
        __import__(__name__ + _mod, fromlist=["*"])
    except Exception:  # pragma: no cover - optional deps may be missing
        pass
