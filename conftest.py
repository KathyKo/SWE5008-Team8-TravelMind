"""
Root conftest.py — loaded by pytest before any test collection begins.

Stubs out uuid_utils (a PyO3 C extension) to prevent a double-initialisation
crash that occurs when pytest-cov's import tracer is active.  Without this
stub, importing langchain_core under coverage causes:
    ImportError: PyO3 modules compiled for CPython 3.8 or older may only be
                 initialized once per interpreter process
"""
import sys
from unittest.mock import MagicMock

for _mod in ("uuid_utils", "uuid_utils.compat", "uuid_utils._uuid_utils"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def pytest_configure(config):
    config.addinivalue_line("markers", "red_team_e2e: full e2e red team tests against live service")
