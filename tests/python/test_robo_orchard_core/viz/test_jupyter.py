# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import pytest

from robo_orchard_core.viz.jupyter.base_viz import BaseIpyViz, DomEvent
from robo_orchard_core.viz.jupyter.ipy_cam import IpyFPVCameraViz
from robo_orchard_core.viz.jupyter.virtual_display import IpyVirtualDisplay


class DummyClosable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyOutput(DummyClosable):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyViz(BaseIpyViz):
    def _on_event(self, event):
        return None


class DummyExecutor:
    def __init__(self):
        self.shutdown_args = None

    def shutdown(self, wait: bool, cancel_futures: bool = False):
        self.shutdown_args = (wait, cancel_futures)


class DummyMss:
    def __init__(self):
        self.exited = False

    def __exit__(self, exc_type, exc, tb):
        self.exited = True


class DummyPyAutoGui:
    def __init__(self):
        self.calls = []

    def keyDown(self, key, _pause=False):
        self.calls.append(("keyDown", key))

    def keyUp(self, key, _pause=False):
        self.calls.append(("keyUp", key))

    def scroll(self, delta, _pause=False):
        self.calls.append(("scroll", delta))

    def resolution(self):
        return (1000, 500)

    def moveTo(self, x, y, _pause=False):
        self.calls.append(("moveTo", x, y))

    def mouseDown(self, button, _pause=False):
        self.calls.append(("mouseDown", button))

    def mouseUp(self, button, _pause=False):
        self.calls.append(("mouseUp", button))


class TestJupyterViz:
    @staticmethod
    def _make_dom_event(**updates: Any) -> DomEvent:
        event: dict[str, Any] = {
            "target": {},
            "altKey": False,
            "metaKey": False,
            "shiftKey": False,
            "ctrlKey": False,
            "type": "mousemove",
            "event": "",
            "buttons": 0,
            "clientX": 0,
            "clientY": 0,
            "offsetX": 0,
            "offsetY": 0,
            "pageX": 0,
            "pageY": 0,
            "screenX": 0,
            "screenY": 0,
            "x": 0,
            "y": 0,
            "relativeX": 0,
            "relativeY": 0,
            "deltaY": 0,
        }
        event.update(updates)
        return cast(DomEvent, event)

    def test_base_display_uses_runtime_import(self, monkeypatch):
        ipy_display = pytest.importorskip("IPython.display")
        display_calls = []
        monkeypatch.setattr(
            ipy_display,
            "display",
            lambda *args: display_calls.append(args),
        )
        viz = DummyViz.__new__(DummyViz)
        canvas = object()
        output = object()
        viz.canvas = cast(Any, canvas)
        viz.output = output
        viz._closed = False

        BaseIpyViz.display(viz)

        assert display_calls == [(canvas, output)]

    def test_virtual_display_display_renders_first_frame(self, monkeypatch):
        ipy_display = pytest.importorskip("IPython.display")
        display_calls = []
        monkeypatch.setattr(
            ipy_display,
            "display",
            lambda *args: display_calls.append(args),
        )
        viz = IpyVirtualDisplay.__new__(IpyVirtualDisplay)
        canvas = object()
        output = object()
        viz.canvas = cast(Any, canvas)
        viz.output = output
        viz._closed = False
        rendered = []
        viz._render = lambda: rendered.append(True)

        IpyVirtualDisplay.display(viz)

        assert display_calls == [(canvas, output)]
        assert rendered == [True]

    def test_fpv_camera_display_uses_runtime_import(self, monkeypatch):
        ipy_display = pytest.importorskip("IPython.display")
        display_calls = []
        monkeypatch.setattr(
            ipy_display,
            "display",
            lambda *args: display_calls.append(args),
        )
        monkeypatch.setattr(
            "robo_orchard_core.viz.jupyter.ipy_cam.widgets",
            SimpleNamespace(
                HBox=lambda children: ("HBox", tuple(children)),
            ),
            raising=False,
        )

        viz = cast(Any, type("DummyFPV", (), {})())
        viz._description = object()
        viz._camera_info_box = object()
        viz.canvas = object()
        viz.output = object()
        rendered = []
        pose_updates = []
        viz._render = lambda: rendered.append(True)
        viz._on_pose_change = lambda: pose_updates.append(True)

        IpyFPVCameraViz.display(viz)

        assert len(display_calls) == 1
        hbox, canvas, output = display_calls[0]
        assert hbox[0] == "HBox"
        assert canvas is viz.canvas
        assert output is viz.output
        assert rendered == [True]
        assert pose_updates == [True]

    def test_virtual_display_mousedown_moves_before_click(
        self, monkeypatch
    ):
        pyautogui = DummyPyAutoGui()
        monkeypatch.setattr(
            "robo_orchard_core.viz.jupyter.virtual_display.pyautogui",
            pyautogui,
            raising=False,
        )

        viz = IpyVirtualDisplay.__new__(IpyVirtualDisplay)
        viz.output = DummyOutput()
        viz.canvas = cast(
            Any, type("Canvas", (), {"width": 200, "height": 100})()
        )
        viz._closed = False
        viz._left_click = False
        viz._right_click = False
        viz._middle_click = False
        viz._render = lambda: None

        IpyVirtualDisplay._on_event(
            viz,
            self._make_dom_event(
                type="mousedown",
                relativeX=100,
                relativeY=50,
                buttons=1,
            ),
        )

        move_idx = pyautogui.calls.index(("moveTo", 500, 250))
        down_idx = pyautogui.calls.index(("mouseDown", "left"))
        assert move_idx < down_idx

    def test_virtual_display_close_releases_resources(self):
        viz = IpyVirtualDisplay.__new__(IpyVirtualDisplay)
        viz._closed = False
        task: Future[None] = Future()
        executor = DummyExecutor()
        mss = DummyMss()
        event = DummyClosable()
        output = DummyClosable()
        viz._last_render_task = task
        viz._render_executor = cast(Any, executor)
        viz._mss = cast(Any, mss)
        viz.event = event
        viz.output = output

        IpyVirtualDisplay.close(viz)

        assert task.cancelled()
        assert executor.shutdown_args == (False, True)
        assert mss.exited is True
        assert event.closed is True
        assert output.closed is True
        assert viz._closed is True
