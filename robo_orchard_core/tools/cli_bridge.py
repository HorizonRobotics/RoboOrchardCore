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

"""Bridge pydantic-settings leaf commands into Typer command nodes.

Use this module when Typer should own command-tree routing, grouping, and root
help while a ``BaseSettings`` subclass owns leaf option parsing, validation,
configuration sources, defaults, and ``command_impl()`` execution. This keeps
the CLI from growing two command-tree sources of truth while preserving the
configuration expressiveness of existing settings models.

The bridge exists because neither side should replace the other. Typer should
not need to mirror every complex settings field as a generated callback
signature, and pydantic-settings should not continue to own cross-layer
command hierarchy in the unified ``robo-orchard`` entrypoint. The stable
boundary is a Typer leaf that forwards raw leaf arguments to one settings
model, then lets that model validate and run its ``command_impl()``.

The adapter has one public construction path: ``as_typer(...)`` returns a
``typer.Typer`` app for either one executable settings class or an explicit
mapping from public Typer leaf names to executable settings classes. Mapping
iteration order is preserved for help output. A single leaf is represented by
the returned app callback, not by an extra same-named command, so callers can
mount it without producing duplicated paths such as ``run run``.

The bridge is intentionally leaf-oriented. It rejects ``CliSubCommand``
aggregate settings classes, string references, and factory indirection; plugin
modules should keep imports light and defer remote SDK setup, credential reads,
network access, and heavy work to ``command_impl()`` or lower runtime layers.
It does not generate Typer option signatures or expand nested settings fields;
Pydantic settings remains the source of truth for leaf parameters.

Adapter leaves parse only the current Typer ``ctx.args`` and do not re-read
global ``sys.argv``. The generated leaves own the permissive Typer context
settings needed to forward raw args to pydantic-settings; root apps, plugin
groups, and ordinary Typer commands should keep Typer's default argument
checking so command-tree spelling errors are not hidden. Public help and
examples should use kebab-case; legacy snake_case options are normalized only
at this adapter boundary.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Protocol, cast

import typer
from pydantic_settings import BaseSettings, CliApp, CliSettingsSource
from typer.core import TyperGroup

LEAF_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


class SettingsCommand(Protocol):
    """Runtime protocol for settings models that can execute a CLI leaf.

    Settings classes adapted by ``PydanticSettingsTyperAdapter`` must be
    ``BaseSettings`` subclasses and expose this method. The adapter owns
    argument parsing, then delegates command behavior to ``command_impl`` on
    the parsed settings instance.
    """

    def command_impl(self) -> object:
        """Run the command represented by the parsed settings instance.

        Returns:
            object: Command-specific result. The adapter does not inspect the
                return value.
        """


class PydanticSettingsTyperAdapter:
    """Adapt Pydantic settings classes into Typer command nodes.

    Use this adapter when Typer should own the command tree and plugin loading
    while ``pydantic-settings`` owns option parsing, validation, and command
    execution. The adapter supports a single executable settings class as a
    leaf command, or an explicit mapping from Typer leaf names to executable
    settings classes.
    """

    def as_typer(
        self,
        command: type[BaseSettings] | Mapping[str, type[BaseSettings]],
        *,
        prog: str,
        description: str | None = None,
    ) -> typer.Typer:
        """Build a Typer app from a settings leaf or explicit leaf mapping.

        Args:
            command (type[BaseSettings] | Mapping[str, type[BaseSettings]]):
                A ``BaseSettings`` subclass with ``command_impl()``, or an
                explicit mapping from Typer leaf command name to such a class.
            prog (str): Full CLI program path shown by Pydantic settings help.
            description (str | None, optional): Help text for the returned
                Typer app. For a single leaf, default is the settings class
                docstring. For a mapping group, default is None.

        Returns:
            A Typer app that runs the provided settings model leaves.

        Raises:
            TypeError: If ``command`` or mapping values are not executable
                settings classes, or if a settings class contains
                ``CliSubCommand``.
            ValueError: If a command mapping is empty.
        """
        if isinstance(command, Mapping):
            return self._mapping_as_typer(
                command,
                prog=prog,
                description=description,
            )

        settings_cls = self._validate_settings_cls(command)
        return self._leaf_as_typer(
            settings_cls,
            prog=prog,
            description=description,
        )

    def _mapping_as_typer(
        self,
        command: Mapping[str, type[BaseSettings]],
        *,
        prog: str,
        description: str | None,
    ) -> typer.Typer:
        if not command:
            raise ValueError("Pydantic settings command mapping is empty.")

        group_app = typer.Typer(cls=_OrderedTyperGroup, help=description)
        for leaf_name, settings_cls in command.items():
            if not isinstance(leaf_name, str) or not leaf_name:
                raise TypeError(
                    "Pydantic settings command mapping keys must be "
                    "non-empty strings."
                )
            validated_cls = self._validate_settings_cls(settings_cls)
            leaf_prog = f"{prog} {leaf_name}"
            leaf_help = _settings_help(validated_cls)

            group_app.command(
                name=leaf_name,
                help=leaf_help,
                context_settings=LEAF_CONTEXT_SETTINGS,
                add_help_option=False,
            )(self._make_leaf_callback(validated_cls, leaf_prog))

        return group_app

    def _leaf_as_typer(
        self,
        settings_cls: type[SettingsCommand],
        *,
        prog: str,
        description: str | None,
    ) -> typer.Typer:
        leaf_app = typer.Typer(
            cls=_SettingsLeafGroup,
            invoke_without_command=True,
            no_args_is_help=False,
            context_settings=LEAF_CONTEXT_SETTINGS,
            add_help_option=False,
            help=description or _settings_help(settings_cls),
        )

        @leaf_app.callback(
            invoke_without_command=True,
            context_settings=LEAF_CONTEXT_SETTINGS,
            add_help_option=False,
        )
        def _leaf(ctx: typer.Context) -> None:
            self._run_settings_leaf(
                settings_cls,
                prog=prog,
                raw_args=list(ctx.args),
            )

        return leaf_app

    def _make_leaf_callback(
        self,
        settings_cls: type[SettingsCommand],
        prog: str,
    ):
        def _leaf(ctx: typer.Context) -> None:
            self._run_settings_leaf(
                settings_cls,
                prog=prog,
                raw_args=list(ctx.args),
            )

        return _leaf

    def _validate_settings_cls(
        self,
        settings_cls: object,
    ) -> type[SettingsCommand]:
        if not isinstance(settings_cls, type) or not issubclass(
            settings_cls,
            BaseSettings,
        ):
            raise TypeError(
                "Pydantic settings commands must be BaseSettings subclasses."
            )

        base_settings_cls = cast(type[BaseSettings], settings_cls)
        if _has_cli_subcommand_field(base_settings_cls):
            raise TypeError(
                "CliSubCommand aggregate settings classes are not supported "
                "by PydanticSettingsTyperAdapter."
            )

        if not callable(getattr(settings_cls, "command_impl", None)):
            raise TypeError(
                "Pydantic settings commands must define command_impl()."
            )

        return cast(type[SettingsCommand], settings_cls)

    def _run_settings_leaf(
        self,
        settings_cls: type[SettingsCommand],
        *,
        prog: str,
        raw_args: list[str],
    ) -> None:
        base_settings_cls = cast(type[BaseSettings], settings_cls)
        normalized_args = _normalize_legacy_snake_options(raw_args)
        cli_settings = CliSettingsSource(
            base_settings_cls,
            cli_prog_name=prog,
            cli_parse_args=normalized_args,
            cli_enforce_required=True,
            cli_avoid_json=True,
            cli_kebab_case=True,
        )
        settings = cast(
            SettingsCommand,
            CliApp.run(
                base_settings_cls,
                cli_args=normalized_args,
                cli_settings_source=cli_settings,
            ),
        )
        settings.command_impl()


def _settings_help(settings_cls: type[SettingsCommand]) -> str | None:
    doc = settings_cls.__doc__
    if doc is None:
        return None
    stripped = doc.strip()
    return stripped or None


def _has_cli_subcommand_field(settings_cls: type[BaseSettings]) -> bool:
    for field in settings_cls.model_fields.values():
        if any(
            getattr(item, "__name__", "") == "_CliSubCommand"
            for item in field.metadata
        ):
            return True
    return False


def _normalize_legacy_snake_options(raw_args: list[str]) -> list[str]:
    """Accept legacy snake-case option spellings under kebab-case help."""
    normalized_args: list[str] = []
    for arg in raw_args:
        if not arg.startswith("--") or arg == "--":
            normalized_args.append(arg)
            continue

        option, separator, value = arg.partition("=")
        normalized_option = f"--{option[2:].replace('_', '-')}"
        normalized_args.append(
            f"{normalized_option}{separator}{value}"
            if separator
            else normalized_option
        )
    return normalized_args


class _SettingsLeafGroup(TyperGroup):
    """Typer group that forwards all leaf args to the callback."""

    def parse_args(self, ctx, args):
        ctx.args = list(args)
        return []


class _OrderedTyperGroup(TyperGroup):
    """Typer group that keeps command insertion order in help output."""

    def list_commands(self, ctx):
        return list(self.commands)
