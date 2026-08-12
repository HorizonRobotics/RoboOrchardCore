# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

import importlib
import logging

import robo_orchard_core.utils.logging as logging_utils


class TestLoggerManager:
    def test_no_default_handler_when_root_has_handlers(self, monkeypatch):
        module = importlib.reload(logging_utils)
        monkeypatch.setattr(
            logging.Logger,
            "hasHandlers",
            lambda _self: True,
        )

        manager = module.LoggerManager(handlers=[])
        assert len(manager.get_logger().handlers) == 0
        assert manager.get_logger().propagate is True

    def test_add_default_handler_when_no_root_handlers(self, monkeypatch):
        module = importlib.reload(logging_utils)
        monkeypatch.setattr(
            logging.Logger,
            "hasHandlers",
            lambda _self: False,
        )

        manager = module.LoggerManager(handlers=[])
        assert len(manager.get_logger().handlers) == 1
        assert isinstance(
            manager.get_logger().handlers[0], logging.StreamHandler
        )
        assert manager.get_logger().propagate is False

    def test_set_handlers_updates_output_ownership(self, monkeypatch):
        """Handler changes should switch between manager and root output."""
        module = importlib.reload(logging_utils)
        monkeypatch.setattr(
            logging.Logger,
            "hasHandlers",
            lambda _self: True,
        )

        manager = module.LoggerManager(handlers=[])
        logger = manager.get_logger()
        assert logger.propagate is True

        manager.set_handlers([logging.NullHandler()])
        assert logger.propagate is False

        manager.set_handlers([])
        assert logger.propagate is True
