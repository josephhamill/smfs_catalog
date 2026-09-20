# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
An AnalysisWorker stand-in for windows that cannot be built without one.

RawCurveWindow follows a worker's playhead and has no timeline of its own, so
the worker is a constructor argument rather than an option. Any test that
builds the window needs something to pass, including tests with nothing to say
about analysis — which is why this is one class shared between modules rather
than a double per test file.

It emits nothing on its own: a test that wants a playhead move emits
`playhead_changed` itself, so the moment is the test's to choose.
"""
from PyQt6.QtCore import QObject, pyqtSignal


class FakeWorker(QObject):
    playhead_changed = pyqtSignal(int)
    queue_empty = pyqtSignal()
    file_done = pyqtSignal(int, str, bool)
    file_error = pyqtSignal(int, str)
    data_unavailable = pyqtSignal(int, str, str)
    paused_changed = pyqtSignal(bool)
    direction_changed = pyqtSignal(int)
    throttle_changed = pyqtSignal(int)
    queue_changed = pyqtSignal()

    def queue_ids(self):
        return []

    def playhead(self):
        return None

    def throttle_ms(self):
        return 0

    def is_paused(self):
        return True

    def direction(self):
        return 1

    def set_paused(self, _paused):
        pass

    def set_direction(self, _direction):
        pass

    def set_throttle_ms(self, _ms):
        pass

    def notify_work_available(self):
        pass
