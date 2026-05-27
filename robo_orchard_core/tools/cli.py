# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""Root command assembly for the ``robo-orchard`` console script.

This module owns the public Typer command tree and extension discovery for the
root executable. The CLI layer is an operator/developer-facing entrypoint
layer: it should stay thin, route commands, and leave reusable runtime logic in
the feature modules that own that behavior.

This split is deliberate: Typer is a better fit for command trees, grouping,
plugin assembly, and top-level CLI user experience, while pydantic-settings is
a better fit for complex leaf parameters, environment/config-source merging,
validation, and executable settings models. The root CLI should therefore
provide one coherent command surface without becoming the source of truth for
every leaf command's configuration schema.

The overall architecture is layered as follows: the installed
``robo-orchard`` console script calls this root Typer app; the root app loads
core-owned built-ins and package-provided Typer extension apps; each extension
app owns its command groups and leaf commands; complex leaf commands may use
``robo_orchard_core.tools.cli_bridge`` to hand raw leaf arguments to a
Pydantic settings model, whose ``command_impl()`` runs the actual operation.
This file owns only the root assembly and extension-loading boundary.

Feature packages should expose Typer apps through the canonical
``robo_orchard.cli`` entry point group instead of patching this module. The
root CLI mounts those apps by entry point name and does not import feature
implementations directly. Entry points are loaded in deterministic name order;
``robo_orchard.cli`` is loaded before the legacy ``robo_orchard.plugins``
group, and an already loaded command name wins with a warning instead of being
silently overwritten. Duplicate-name warnings identify both the spec that stays
loaded and the spec that was skipped.

Built-in commands belong here only when the command name is core-owned or must
remain visible without optional extras. Register those built-ins by
``module:object`` spec plus required-module checks so root help stays stable;
when requirements are missing, the root CLI mounts a placeholder that reports
the install requirement. Built-in modules and plugin modules should remain
import-light because root help loads their Typer apps during command discovery.

Extension load failures warn and continue by default so one broken plugin does
not break the whole CLI. Call ``load_cli_extensions(..., strict=True)`` when
startup validation should fail fast instead.
"""

import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from importlib.util import find_spec

import typer

CLI_ENTRY_POINT_GROUP = "robo_orchard.cli"
LEGACY_PLUGIN_ENTRY_POINT_GROUP = "robo_orchard.plugins"


@dataclass(frozen=True)
class BuiltinCliExtension:
    """Describe a built-in CLI extension loaded by the root command.

    The extension is loaded from ``target`` only after every module in
    ``required_modules`` is importable. This keeps optional commands visible
    in root help without importing their optional runtime dependencies during
    base CLI startup.

    The fields store the entry-point style ``module:object`` target, root help
    text, optional import checks, and the install requirement shown when an
    optional built-in command is unavailable.
    """

    target: str
    help: str
    required_modules: tuple[str, ...] = ()
    requirement: str | None = None


BUILTIN_CLI_EXTENSIONS = {
    "file-server": BuiltinCliExtension(
        target="robo_orchard_core.tools.simple_file_server:cli_app",
        help="Simple HTTP file server.",
        required_modules=("fastapi", "aiofiles", "uvicorn"),
        requirement="robo_orchard_core[tools]",
    ),
}


def create_app(load_extensions: bool = True) -> typer.Typer:
    """Create the RoboOrchard root CLI app.

    Args:
        load_extensions (bool, optional): Whether to load built-in and
            entry-point CLI extensions. Default is True.

    Returns:
        typer.Typer: Root Typer application for the ``robo-orchard`` command.
    """
    cli_app = typer.Typer(
        help=(
            "Robo Orchard Core CLI - A unified toolset for robotics "
            "development."
        ),
        add_completion=False,
    )

    if load_extensions:
        load_cli_extensions(cli_app)

    return cli_app


def load_cli_extensions(
    cli_app: typer.Typer,
    *,
    strict: bool = False,
) -> None:
    """Mount built-in and plugin Typer apps onto a root CLI app.

    Built-in commands are loaded first. If a built-in command has missing
    optional dependencies, a placeholder command is mounted so root help stays
    stable and invocation reports the missing extra. Entry points from
    ``robo_orchard.cli`` are then loaded before legacy
    ``robo_orchard.plugins`` entry points.

    Args:
        cli_app (typer.Typer): Root application to mutate by adding command
            groups.
        strict (bool, optional): Whether loading failures should raise
            ``RuntimeError`` instead of printing warnings and continuing.
            Default is False.

    Raises:
        RuntimeError: If ``strict`` is True and a built-in or plugin extension
            cannot be loaded.
    """
    loaded_specs: dict[str, str] = {}
    for builtin_name, builtin_extension in BUILTIN_CLI_EXTENSIONS.items():
        missing_modules = _missing_modules(builtin_extension.required_modules)
        if missing_modules:
            cli_app.add_typer(
                _missing_dependency_app(
                    builtin_name,
                    builtin_extension,
                    missing_modules=missing_modules,
                ),
                name=builtin_name,
            )
            loaded_specs[builtin_name] = builtin_extension.target
            continue

        try:
            cli_app.add_typer(
                _load_cli_app_from_spec(builtin_extension.target),
                name=builtin_name,
            )
            loaded_specs[builtin_name] = builtin_extension.target
        except ImportError as e:
            if strict:
                message = (
                    f"Warning: Failed to load built-in CLI extension "
                    f"'{builtin_name}': {e}"
                )
                raise RuntimeError(message) from e
            cli_app.add_typer(
                _missing_dependency_app(
                    builtin_name,
                    builtin_extension,
                    missing_modules=(),
                ),
                name=builtin_name,
            )
            loaded_specs[builtin_name] = builtin_extension.target
        except Exception as e:
            message = (
                f"Warning: Failed to load built-in CLI extension "
                f"'{builtin_name}': {e}"
            )
            if strict:
                raise RuntimeError(message) from e
            print(message, file=sys.stderr)

    groups = (
        (CLI_ENTRY_POINT_GROUP, False),
        (LEGACY_PLUGIN_ENTRY_POINT_GROUP, True),
    )

    for group, is_legacy in groups:
        discovered_extensions = sorted(
            entry_points(group=group),
            key=lambda item: item.name,
        )
        for entry_point in discovered_extensions:
            label = "legacy plugin" if is_legacy else "CLI extension"
            loaded_spec = loaded_specs.get(entry_point.name)
            entry_point_spec = getattr(entry_point, "value", None)
            if loaded_spec is not None:
                if entry_point_spec == loaded_spec:
                    continue
                print(
                    f"Warning: Skipping {label} '{entry_point.name}' from "
                    f"{_display_spec(entry_point_spec)} because command is "
                    f"already loaded from {_display_spec(loaded_spec)}.",
                    file=sys.stderr,
                )
                continue

            try:
                extension_app = entry_point.load()
                if not isinstance(extension_app, typer.Typer):
                    raise TypeError(
                        "entry point did not load to a typer.Typer instance"
                    )
                cli_app.add_typer(extension_app, name=entry_point.name)
                loaded_specs[entry_point.name] = entry_point_spec or ""
            except Exception as e:
                message = (
                    f"Warning: Failed to load {label} "
                    f"'{entry_point.name}': {e}"
                )
                if strict:
                    raise RuntimeError(message) from e
                print(message, file=sys.stderr)


def _load_cli_app_from_spec(spec: str) -> typer.Typer:
    """Load a Typer app from an entry-point style ``module:object`` spec."""
    module_name, object_name = spec.split(":", maxsplit=1)
    extension_app = getattr(import_module(module_name), object_name)
    if not isinstance(extension_app, typer.Typer):
        raise TypeError(f"CLI extension {spec!r} is not a Typer app")
    return extension_app


def _display_spec(spec: str | None) -> str:
    return repr(spec) if spec else "<unknown>"


def _missing_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in module_names if find_spec(name) is None)


def _missing_dependency_app(
    command_name: str,
    builtin_extension: BuiltinCliExtension,
    *,
    missing_modules: tuple[str, ...],
) -> typer.Typer:
    """Build a placeholder app for built-ins gated by optional deps."""
    placeholder_app = typer.Typer(
        help=builtin_extension.help,
        invoke_without_command=True,
        no_args_is_help=False,
        add_completion=False,
    )

    @placeholder_app.callback(invoke_without_command=True)
    def _missing_dependency() -> None:
        requirement = builtin_extension.requirement or "required extras"
        missing = (
            f" Missing modules: {', '.join(missing_modules)}."
            if missing_modules
            else ""
        )
        typer.echo(
            f"Command '{command_name}' requires `{requirement}`.{missing}",
            err=True,
        )
        raise typer.Exit(1)

    return placeholder_app


def load_plugins(cli_app: typer.Typer) -> None:
    """Load CLI extensions for callers using the old plugin helper name.

    Args:
        cli_app (typer.Typer): Root application to mutate by adding command
            groups.
    """
    load_cli_extensions(cli_app)


app = create_app()


def main() -> None:
    """Run the installed ``robo-orchard`` console script."""
    app()


if __name__ == "__main__":
    main()
