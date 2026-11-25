"""provenance.sklearn_version(): one shared copy, narrow exception handling."""

import ast
import re
import sys
import types
from pathlib import Path

import pytest

from smfs_catalog import provenance

_PKG = Path(__file__).resolve().parent.parent / "smfs_catalog"
_WINDOWS = ["gmm_fit_window.py", "pca_window.py"]


def test_returns_installed_sklearn_version():
    import sklearn
    assert provenance.sklearn_version() == str(sklearn.__version__)


def test_unknown_when_sklearn_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)  # import -> ImportError
    assert provenance.sklearn_version() == "unknown"


def test_unknown_when_sklearn_has_no_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    assert provenance.sklearn_version() == "unknown"


def test_other_exceptions_propagate(monkeypatch):
    class Broken(types.ModuleType):
        @property
        def __version__(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "sklearn", Broken("sklearn"))
    with pytest.raises(RuntimeError):
        provenance.sklearn_version()


@pytest.mark.parametrize("name", _WINDOWS)
def test_window_has_no_private_copy(name):
    src = (_PKG / name).read_text(encoding="utf-8")
    assert not re.search(r"\b_sklearn_version\b", src)


def _tree(name):
    return ast.parse((_PKG / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", _WINDOWS)
def test_window_imports_sklearn_version_from_provenance(name):
    imported = any(
        isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "provenance"
        and any(a.name == "sklearn_version" and a.asname is None for a in n.names)
        for n in ast.walk(_tree(name)))
    assert imported, f"{name} must do `from .provenance import sklearn_version`"


@pytest.mark.parametrize("name", _WINDOWS)
def test_window_does_not_define_its_own_sklearn_version(name):
    defs = [n.name for n in ast.walk(_tree(name))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.lstrip("_") == "sklearn_version"]
    assert defs == []
