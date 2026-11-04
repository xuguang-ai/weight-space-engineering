"""Shared test fixtures. Runtime outputs go to pytest's tmp_path only."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from wse.api import Engine

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = REPO_ROOT / "examples" / "signed_sum" / "build_fixture.py"


@pytest.fixture(scope="session")
def signed_sum() -> ModuleType:
    """The signed-sum fixture builder (operator/test use only)."""
    spec = importlib.util.spec_from_file_location("signed_sum_build_fixture", FIXTURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundles(tmp_path: Path, signed_sum: ModuleType) -> dict[str, Path]:
    return signed_sum.build(tmp_path / "bundles", with_reference=True)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def engine(workspace: Path):
    with Engine(workspace) as managed:
        yield managed
