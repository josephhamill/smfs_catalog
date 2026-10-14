import ast
from pathlib import Path


PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_spinboxes_do_not_recompute_on_live_value_changes():
    source = (PKG / "display_roi.py").read_text(encoding="utf-8")
    assert "valueChanged.connect(self._preview" not in source
    assert "editingFinished.connect(self._commit" in source


def test_roi_fit_panel_draws_the_signal_the_fits_were_made_on():
    source = _function_source(PKG / "display_roi.py", "_draw_fx")
    assert "fit_deflection(" in source
    assert "defl_corr = (np.asarray(low_retr" not in source


def test_wlc_residual_uses_the_signal_the_fits_were_made_on():
    source = _function_source(PKG / "wlc_view_window.py", "_load_and_fit")
    assert "fit_deflection(" in source
    assert "defl_corr = (curve.defl_retr" not in source
    assert "defl_corr = (res.dc.low_retr" not in source


def test_isoforce_geometry_uses_the_same_trace_as_segment_fitting():
    source = _function_source(PKG / "isoforce_window.py", "_load_and_mark")
    assert "res.dc.low_retr" in source
    assert "defl_corr = (curve.defl_retr" not in source
