"""
Shared fixtures and import bootstrap for Agent4 (Debate) tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy.orm import DeclarativeBase

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 在加载 agents.db.database 之前注册占位，避免真实 create_engine（依赖 psycopg2）
_FAKE_DB = "agents.db.database"
if _FAKE_DB not in sys.modules:
    _db_mod = types.ModuleType(_FAKE_DB)

    class _StubBase(DeclarativeBase):
        pass

    _db_mod.Base = _StubBase
    _db_mod.SessionLocal = MagicMock()
    _db_mod.engine = MagicMock()

    def _get_db():
        yield MagicMock()

    _db_mod.get_db = _get_db
    sys.modules[_FAKE_DB] = _db_mod

# Stub optional security deps not installed in all environments
for _mod in (
    "llm_guard",
    "llm_guard.input_scanners",
    "llm_guard.output_scanners",
    "tools.security.llm_guard_scanner",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
