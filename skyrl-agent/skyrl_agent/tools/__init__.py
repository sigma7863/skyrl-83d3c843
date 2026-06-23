from .base import TOOL_REGISTRY, BaseTool
from .finish import FinishTool
from .em_finish import EMFinishTool

# The remaining tools pull in heavy optional dependencies (faiss, tevatron,
# openai client, browser/search stacks). Import them lazily so that minimal
# setups using only the lightweight tools (e.g. ``finish``) do not require the
# full optional dependency tree. Any import failure is swallowed; the tool
# simply will not be registered if its dependencies are unavailable.
for _mod in (
    ".sandbox_fusion",
    ".search_engine",
    ".youcom_search_engine",
    ".web_browser",
    ".local_search",
    ".next_memagent",
    ".search",
):
    try:
        __import__(__name__ + _mod, fromlist=["*"])
    except Exception:  # pragma: no cover - optional deps may be missing
        pass
