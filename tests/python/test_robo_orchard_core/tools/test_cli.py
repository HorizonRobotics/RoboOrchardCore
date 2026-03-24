# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import typer
from typer.testing import CliRunner

import robo_orchard_core.tools.cli as cli_module
import robo_orchard_core.tools.simple_file_server as file_server_module
import robo_orchard_core.utils.network as network_module

runner = CliRunner()


class DummyEntryPoint:
    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class TestCli:
    def test_help_lists_file_server(self):
        result = runner.invoke(cli_module.app, ["--help"])

        assert result.exit_code == 0
        assert "file-server" in result.output

    def test_file_server_uses_free_port(self, monkeypatch):
        called = {}

        monkeypatch.setattr(network_module, "find_free_port", lambda: 4321)
        monkeypatch.setattr(
            file_server_module,
            "start_server",
            lambda host, port, directory: called.update(
                host=host,
                port=port,
                directory=directory,
            ),
        )

        result = runner.invoke(
            cli_module.app,
            ["file-server", "--host", "0.0.0.0", "--dir", "demo"],
        )

        assert result.exit_code == 0
        assert called == {
            "host": "0.0.0.0",
            "port": 4321,
            "directory": "demo",
        }

    def test_load_plugins_warns_on_failure(self, monkeypatch, capsys):
        def broken_loader():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: [DummyEntryPoint("broken", broken_loader)],
        )

        cli_app = typer.Typer()
        cli_module.load_plugins(cli_app)

        captured = capsys.readouterr()
        assert "Failed to load plugin 'broken': boom" in captured.out
