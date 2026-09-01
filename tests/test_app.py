"""The Tk app's own logic: undo, the auto-recognize timer, the result pump.

These drive the handlers on a real window (withdrawn, never mapped).
``mainloop`` is never entered, so the ``after`` callbacks scheduled in
``__init__`` — including the one that loads a recognizer — never fire, and no
model is touched.

One window serves the whole module, reset between tests. Building a Tk root is
the slow and failure-prone part, and some Tcl installs get unreliable when a
process creates and destroys a dozen of them.

Skipped wherever Tk cannot open a display, which includes a headless SSH session
on the Pi. Run them on the machine you edit on.
"""

import pytest

tk = pytest.importorskip("tkinter")

from handwriting_app.config import AppConfig  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from handwriting_app.app import HandwritingApp

    try:
        app = HandwritingApp(AppConfig())
    except tk.TclError as exc:  # no display, or a broken Tcl install
        pytest.skip(f"Tk unavailable: {exc}")
    app.withdraw()
    yield app
    app.destroy()


@pytest.fixture
def app(window):
    """The shared window, wound back to a freshly-started state."""
    window._cancel_pending_auto()
    window.canvas.clear()
    window.output.delete("1.0", "end")
    window.auto_var.set(False)
    window._clear_pending = False
    window._partial = ""
    window._busy = False
    window._recognize_started = None
    window._set_status("")
    return window


def _write(window, *x_starts):
    """Put one stroke per x position on the pad, as drawing would."""
    for x0 in x_starts:
        stroke = window.canvas.ink.start_stroke()
        for x, y in ((x0, 0.0), (x0 + 20.0, 40.0), (x0 + 40.0, 0.0)):
            stroke.add(x, y)
    window.canvas.redraw()


def _status(window) -> str:
    return window.status.cget("text")


def _output(window) -> str:
    return window.output.get("1.0", "end-1c")


# -- undo -----------------------------------------------------------------


def test_undo_removes_only_the_last_stroke(app):
    _write(app, 0, 100, 200)
    app._undo()
    assert len(app.canvas.ink.strokes) == 2
    assert "Undid" in _status(app)


def test_undo_on_an_empty_pad_says_so(app):
    app._undo()
    assert _status(app) == "Nothing to undo"


def test_undo_after_a_result_clears_the_pad_instead(app):
    """The ink on screen has already become text. Undoing into it would leave a
    fragment of a finished line to be recognized a second time."""
    _write(app, 0, 100)
    app._clear_pending = True
    app._undo()
    assert app.canvas.ink.is_empty
    assert app._clear_pending is False
    assert _status(app) == "Pad cleared"


def test_undo_restarts_the_auto_recognize_countdown(app):
    app.auto_var.set(True)
    _write(app, 0, 100)
    app._on_stroke_end()
    first = app._pending_auto
    assert first is not None

    app._undo()
    # The old timer would have fired on ink that no longer exists.
    assert app._pending_auto is not None
    assert app._pending_auto != first


def test_undoing_the_last_stroke_leaves_no_timer_running(app):
    app.auto_var.set(True)
    _write(app, 0)
    app._on_stroke_end()
    app._undo()
    assert app.canvas.ink.is_empty
    assert app._pending_auto is None


# -- the pad between recognitions -----------------------------------------


def test_writing_again_wipes_ink_that_was_already_read(app):
    _write(app, 0)
    app._clear_pending = True
    app._on_stroke_start()
    assert app.canvas.ink.is_empty
    assert app._clear_pending is False


def test_clearing_the_pad_cancels_a_pending_recognition(app):
    app.auto_var.set(True)
    _write(app, 0)
    app._on_stroke_end()
    app._clear_pad()
    assert app._pending_auto is None


# -- the result pump -------------------------------------------------------


def test_a_result_is_appended_and_reported(app):
    app._handle_result("result", ("hello world", ""))
    assert _output(app) == "hello world"
    assert "hello world" in _status(app)


def test_a_result_reports_what_ink_cleanup_removed(app):
    """Cleanup deletes strokes the user can still see, so it has to be said."""
    app._handle_result("result", ("hello", "1 drag cut"))
    assert "1 drag cut" in _status(app)


def test_an_empty_result_does_not_write_an_empty_line(app):
    app._handle_result("result", ("", ""))
    assert _output(app) == ""
    assert "No text recognized" in _status(app)


def test_successive_results_are_separated(app):
    app._handle_result("result", ("hello", ""))
    app._handle_result("result", ("world", ""))
    assert _output(app) == "hello world"


# -- the live decode preview ----------------------------------------------


def test_partial_text_shows_up_in_the_status_line(app):
    app._handle_result("partial", "the qui")
    assert "the qui" in app._preview()


def test_a_long_preview_is_trimmed_from_the_front(app):
    app._handle_result("partial", "x" * 400)
    preview = app._preview()
    assert len(preview) < 80 and preview.lstrip().startswith("▸ …")


def test_the_preview_is_dropped_once_the_result_lands(app):
    app._handle_result("partial", "hell")
    app._handle_result("result", ("hello", ""))
    assert app._preview() == ""


def test_there_is_no_preview_before_anything_is_decoded(app):
    assert app._preview() == ""
