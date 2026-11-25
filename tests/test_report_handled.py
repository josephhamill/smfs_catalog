"""qt_utils.report_handled: log the caught exception, then show a one-line message."""
import inspect

import pytest

from smfs_catalog import crashlog, qt_utils
from smfs_catalog.rawcurve_window import RawCurveWindow


@pytest.fixture
def events(monkeypatch):
    """Ordered record of log_handled calls; the sink is replaced, nothing else."""
    log = []
    monkeypatch.setattr(
        crashlog, "log_handled", lambda ctx, exc: log.append(("log", ctx, exc))
    )
    return log


def _show_into(events):
    return lambda text: events.append(("show", text))


def test_worked_example_text(events):
    shown = []
    qt_utils.report_handled(shown.append, "Load curve", FileNotFoundError("x.ibw"))
    assert shown == ["Load curve failed: FileNotFoundError: x.ibw"]


def test_message_cut_to_120_chars(events):
    shown = []
    qt_utils.report_handled(shown.append, "Fit", ValueError("a" * 200))
    prefix = "Fit failed: ValueError: "
    assert shown == [prefix + "a" * 120]
    assert len(shown[0]) - len(prefix) == 120


def test_logs_once_with_same_context_and_exception_before_show(events):
    exc = RuntimeError("boom")
    qt_utils.report_handled(_show_into(events), "Save", exc)
    assert [e[0] for e in events] == ["log", "show"]
    kind, ctx, logged = events[0]
    assert ctx == "Save"
    assert logged is exc


def test_empty_message_keeps_trailing_separator(events):
    shown = []
    qt_utils.report_handled(shown.append, "Fit", ValueError())
    assert shown == ["Fit failed: ValueError: "]


def test_show_overlay_error_delegates_to_report_handled():
    src = inspect.getsource(RawCurveWindow._show_overlay_error)
    assert "report_handled" in src
    assert "[:120]" not in src
