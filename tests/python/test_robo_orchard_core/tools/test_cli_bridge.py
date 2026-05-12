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

from __future__ import annotations
import sys
from typing import Any, ClassVar, cast

import pytest
import typer
from pydantic import Field
from pydantic_settings import CliSubCommand
from typer.testing import CliRunner

from robo_orchard_core.tools.cli_bridge import (
    PydanticSettingsTyperAdapter,
)
from robo_orchard_core.utils.cli import SettingConfig

runner = CliRunner()


class DemoConfig(SettingConfig):
    """Run the demo command."""

    calls: ClassVar[list[tuple[str, int]]] = []

    name: str = Field(description="Name to use.")
    count: int = Field(default=1, description="Repeat count.")

    def command_impl(self) -> None:
        self.__class__.calls.append((self.name, self.count))


class AliasCompatConfig(SettingConfig):
    """Command with legacy snake-case options."""

    calls: ClassVar[list[tuple[str, bool]]] = []

    some_name: str = Field(description="Name with an underscore.")
    execute: bool = Field(default=True, description="Whether to execute.")

    def command_impl(self) -> None:
        self.__class__.calls.append((self.some_name, self.execute))


class AlphaConfig(SettingConfig):
    """Alpha command."""

    calls: ClassVar[list[str]] = []

    value: str = Field(description="Alpha value.")

    def command_impl(self) -> None:
        self.__class__.calls.append(self.value)


class ZuluConfig(SettingConfig):
    """Zulu command."""

    calls: ClassVar[list[str]] = []

    value: str = Field(description="Zulu value.")

    def command_impl(self) -> None:
        self.__class__.calls.append(self.value)


class ChildConfig(SettingConfig):
    value: str

    def command_impl(self) -> None:
        pass


class MissingCommandImplConfig(SettingConfig):
    value: str


class AggregateConfig(SettingConfig):
    child: CliSubCommand[ChildConfig] = Field(description="Child command.")


class TestPydanticSettingsTyperAdapter:
    def setup_method(self):
        DemoConfig.calls.clear()
        AliasCompatConfig.calls.clear()
        AlphaConfig.calls.clear()
        ZuluConfig.calls.clear()

    def test_single_leaf_app_parses_current_typer_args_only(
        self, monkeypatch
    ):
        adapter = PydanticSettingsTyperAdapter()
        app = typer.Typer()
        app.add_typer(
            adapter.as_typer(
                DemoConfig,
                prog="demo leaf",
                description="Run one demo.",
            ),
            name="leaf",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["polluted", "--name", "wrong", "--count", "99"],
        )

        result = runner.invoke(
            app,
            ["leaf", "--name", "alice", "--count", "3"],
        )

        assert result.exit_code == 0, result.output
        assert DemoConfig.calls == [("alice", 3)]

    def test_single_leaf_help_is_served_by_pydantic_settings(self):
        adapter = PydanticSettingsTyperAdapter()
        app = typer.Typer()
        app.add_typer(
            adapter.as_typer(
                DemoConfig,
                prog="demo leaf",
                description="Run one demo.",
            ),
            name="leaf",
        )

        result = runner.invoke(app, ["leaf", "--help"])

        assert result.exit_code == 0, result.output
        assert "demo leaf" in result.output
        assert "Name to use." in result.output
        assert "Repeat count." in result.output

    def test_leaf_accepts_snake_and_kebab_option_spellings(self):
        adapter = PydanticSettingsTyperAdapter()
        app = typer.Typer()
        app.add_typer(
            adapter.as_typer(
                AliasCompatConfig,
                prog="demo alias",
            ),
            name="alias",
        )

        snake_result = runner.invoke(
            app,
            ["alias", "--some_name", "snake", "--execute", "False"],
        )
        kebab_result = runner.invoke(
            app,
            ["alias", "--some-name", "kebab", "--execute", "False"],
        )

        assert snake_result.exit_code == 0, snake_result.output
        assert kebab_result.exit_code == 0, kebab_result.output
        assert AliasCompatConfig.calls == [
            ("snake", False),
            ("kebab", False),
        ]

    def test_mapping_creates_ordered_leaf_commands(self):
        adapter = PydanticSettingsTyperAdapter()
        app = adapter.as_typer(
            {
                "zulu-command": ZuluConfig,
                "alpha-command": AlphaConfig,
            },
            prog="demo group",
            description="Demo group.",
        )

        help_result = runner.invoke(app, ["--help"])
        run_result = runner.invoke(
            app,
            ["alpha-command", "--value", "payload"],
        )

        assert help_result.exit_code == 0, help_result.output
        zulu_position = help_result.output.find("zulu-command")
        alpha_position = help_result.output.find("alpha-command")
        assert zulu_position < alpha_position
        assert run_result.exit_code == 0, run_result.output
        assert AlphaConfig.calls == ["payload"]
        assert ZuluConfig.calls == []

    def test_rejects_empty_mapping(self):
        adapter = PydanticSettingsTyperAdapter()

        with pytest.raises(ValueError, match="empty"):
            adapter.as_typer({}, prog="demo")

    def test_rejects_non_settings_command(self):
        adapter = PydanticSettingsTyperAdapter()

        with pytest.raises(TypeError, match="BaseSettings"):
            adapter.as_typer(cast(Any, "package.module:Config"), prog="demo")

    def test_rejects_mapping_value_that_is_not_settings_class(self):
        adapter = PydanticSettingsTyperAdapter()

        with pytest.raises(TypeError, match="BaseSettings"):
            adapter.as_typer(cast(Any, {"bad": object}), prog="demo")

    def test_rejects_settings_class_without_command_impl(self):
        adapter = PydanticSettingsTyperAdapter()

        with pytest.raises(TypeError, match="command_impl"):
            adapter.as_typer(MissingCommandImplConfig, prog="demo")

    def test_rejects_cli_subcommand_aggregate(self):
        adapter = PydanticSettingsTyperAdapter()

        with pytest.raises(TypeError, match="CliSubCommand"):
            adapter.as_typer(AggregateConfig, prog="demo")
