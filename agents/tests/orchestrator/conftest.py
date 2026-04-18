import sys
from unittest.mock import MagicMock

# ── Stub heavy dependencies ──────────────────────────────────
# These get pulled in transitively via agents/__init__.py when Python
# imports the `agents` package (which happens when we import agents.graph).
#
# The orchestrator tests only exercise agents.nodes / agents.state / agents.graph
# and mock all remote agent calls, so the real specialist code is never needed.
#
# Under pytest-cov, the uuid_utils C extension (PyO3) crashes if initialised
# twice in the same process. Stubbing the specialist modules at the package
# level prevents the entire langchain_openai → uuid_utils chain from loading.
# ──────────────────────────────────────────────────────────────

_STUB_MODULES = [
    # ML / security libs
    "llm_guard",
    "llm_guard.input_scanners",
    "llm_guard.output_scanners",
    "torch",
    "transformers",
    # Specialist agent modules — prevents agents/__init__.py from
    # importing the real code and triggering heavy transitive deps
    "agents.specialists",
    "agents.specialists.input_guard_agent",
    "agents.specialists.output_guard_agent",
    "agents.specialists.intent_profile",
    "agents.specialists.planner_agent",
    # "agents.specialists.replanner_agent",
    # "agents.specialists.debate_agent",
    "agents.specialists.research_agent",
    "agents.specialists.explainability_agent",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
