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

import logging

import robo_orchard_core.utils.timer as timer_module


class RecordingLogger(logging.Logger):
    def __init__(self):
        super().__init__("timer-test")
        self.records: list[str] = []

    def getChild(self, suffix: str):
        return self

    def debug(self, msg, *args, **kwargs):
        self.records.append(msg)


class TestTimer:
    def test_auto_unit_uses_milliseconds_for_short_durations(
        self, monkeypatch
    ):
        logger = RecordingLogger()
        timer = timer_module.Timer("timer", unit="auto", logger=logger)
        timer._begin = 0.0

        monkeypatch.setattr(timer_module, "perf_counter", lambda: 0.5)

        timer._stop("timer")

        assert logger.records == ["cost 500.000 ms"]

    def test_auto_unit_uses_seconds_for_long_durations(self, monkeypatch):
        logger = RecordingLogger()
        timer = timer_module.Timer("timer", unit="auto", logger=logger)
        timer._begin = 0.0

        monkeypatch.setattr(timer_module, "perf_counter", lambda: 2.0)

        timer._stop("timer")

        assert logger.records == ["cost 2.000 s"]
