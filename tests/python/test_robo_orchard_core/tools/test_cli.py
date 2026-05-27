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

import ast
import inspect
from pathlib import Path

import rtoml
import typer
from typer.testing import CliRunner

import robo_orchard_core.tools.cli as cli_module

runner = CliRunner()


class DummyEntryPoint:
    def __init__(self, name: str, loader, value: str | None = None):
        self.name = name
        self._loader = loader
        self.value = value

    def load(self):
        return self._loader()


class TestCli:
    def test_help_lists_builtin_file_server_without_entry_points(
        self, monkeypatch
    ):
        monkeypatch.setattr(cli_module, "entry_points", lambda group: [])

        result = runner.invoke(cli_module.create_app(), ["--help"])

        assert result.exit_code == 0, result.output
        assert "file-server" in result.output

    def test_help_lists_cli_entry_point(self, monkeypatch):
        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: [
                DummyEntryPoint("external", lambda: _dummy_plugin("served"))
            ]
            if group == "robo_orchard.cli"
            else [],
        )

        result = runner.invoke(cli_module.create_app(), ["--help"])

        assert result.exit_code == 0
        assert "external" in result.output

    def test_cli_module_does_not_import_simple_file_server_directly(self):
        source = inspect.getsource(cli_module)
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

        assert (
            "robo_orchard_core.tools.simple_file_server"
            not in imported_modules
        )

    def test_builtin_file_server_wins_over_same_named_entry_point(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: [
                DummyEntryPoint("file-server", lambda: _dummy_plugin("other"))
            ]
            if group == "robo_orchard.cli"
            else [],
        )

        app = cli_module.create_app()
        captured = capsys.readouterr()
        root_result = runner.invoke(app, ["--help"])
        result = runner.invoke(app, ["file-server", "--help"])

        assert root_result.exit_code == 0, root_result.output
        assert "Simple HTTP file server." in root_result.output
        assert result.exit_code == 0, result.output
        assert "Allow serving files through symlinks." in result.output
        assert "Skipping CLI extension 'file-server' from <unknown>" in (
            captured.err
        )
        assert (
            "already loaded from "
            "'robo_orchard_core.tools.simple_file_server:cli_app'"
            in captured.err
        )

    def test_same_spec_duplicate_entry_point_skips_without_warning(
        self, monkeypatch, capsys
    ):
        target = "robo_orchard_core.tools.simple_file_server:cli_app"

        def fail_if_loaded():
            raise AssertionError("duplicate entry point should be skipped")

        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: [
                DummyEntryPoint("file-server", fail_if_loaded, value=target)
            ]
            if group == "robo_orchard.cli"
            else [],
        )

        app = cli_module.create_app()
        captured = capsys.readouterr()
        result = runner.invoke(app, ["file-server", "--help"])

        assert result.exit_code == 0, result.output
        assert "Allow serving files through symlinks." in result.output
        assert captured.err == ""

    def test_missing_builtin_dependency_keeps_placeholder_command(
        self, monkeypatch
    ):
        def fail_if_loaded(spec):
            raise AssertionError("missing dependencies should skip import")

        monkeypatch.setattr(
            cli_module,
            "_load_cli_app_from_spec",
            fail_if_loaded,
        )
        monkeypatch.setattr(
            cli_module,
            "_missing_modules",
            lambda module_names: ("fastapi",),
        )
        monkeypatch.setattr(cli_module, "entry_points", lambda group: [])

        app = cli_module.create_app()
        help_result = runner.invoke(app, ["--help"])
        command_result = runner.invoke(app, ["file-server"])

        assert help_result.exit_code == 0, help_result.output
        assert "file-server" in help_result.output
        assert command_result.exit_code == 1, command_result.output
        assert "robo_orchard_core[tools]" in command_result.output

    def test_load_cli_extensions_prefers_new_group_on_conflict(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: {
                "robo_orchard.cli": [
                    DummyEntryPoint("shared", lambda: _dummy_plugin("new")),
                ],
                "robo_orchard.plugins": [
                    DummyEntryPoint(
                        "shared", lambda: _dummy_plugin("legacy")
                    ),
                    DummyEntryPoint(
                        "legacy-only",
                        lambda: _dummy_plugin("legacy-only"),
                    ),
                ],
            }[group],
        )

        app = cli_module.create_app()
        captured = capsys.readouterr()
        shared_result = runner.invoke(app, ["shared", "ping"])
        legacy_result = runner.invoke(app, ["legacy-only", "ping"])

        assert shared_result.exit_code == 0
        assert shared_result.output.strip() == "new"
        assert legacy_result.exit_code == 0
        assert legacy_result.output.strip() == "legacy-only"
        assert "Skipping legacy plugin 'shared' from <unknown>" in (
            captured.err
        )
        assert "already loaded from <unknown>" in captured.err

    def test_load_cli_extensions_warns_on_failure_and_continues(
        self, monkeypatch, capsys
    ):
        def broken_loader():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            cli_module,
            "entry_points",
            lambda group: [
                DummyEntryPoint("broken", broken_loader),
                DummyEntryPoint("working", lambda: _dummy_plugin("ok")),
            ]
            if group == "robo_orchard.cli"
            else [],
        )

        app = cli_module.create_app()

        captured = capsys.readouterr()
        result = runner.invoke(app, ["working", "ping"])

        assert result.exit_code == 0
        assert result.output.strip() == "ok"
        assert "Failed to load CLI extension 'broken': boom" in captured.err

    def test_file_server_web_stack_stays_in_tools_extra(self):
        pyproject_path = Path(__file__).parents[4] / "pyproject.toml"
        pyproject = rtoml.load(pyproject_path)
        dependencies = set(pyproject["project"]["dependencies"])
        tools_extra = set(
            pyproject["project"]["optional-dependencies"]["tools"]
        )

        assert "typer" in dependencies
        assert "fastapi" not in dependencies
        assert "aiofiles" not in dependencies
        assert "uvicorn" not in dependencies
        assert {"fastapi", "aiofiles", "uvicorn"} <= tools_extra


def _dummy_plugin(message: str) -> typer.Typer:
    app = typer.Typer(help=f"{message} plugin.")

    @app.command()
    def ping():
        typer.echo(message)

    return app
