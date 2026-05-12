# CLI Plugin Scaffold

Use this scaffold with `.agents/references/cli-guideline.md` when
adding a package CLI app under `robo-orchard`.

Keep only the parts that match the package. Rename modules and command names
to the package's local vocabulary.

## Leaf Command Model

```python
from pydantic import Field

from robo_orchard_core.utils.cli import SettingConfig


class RunConfig(SettingConfig):
    """Run <task>."""

    value: str = Field(description="<value description>.")

    def command_impl(self) -> None:
        """Execute the command after pydantic-settings parsing."""
        ...
```

## Command Map

```python
APP_COMMANDS = {
    "run": RunConfig,
    "version": VersionConfig,
}
```

Use command maps instead of `CliSubCommand` aggregates for canonical Typer
command groups.

## Package CLI App

```python
from robo_orchard_core.tools.cli_bridge import PydanticSettingsTyperAdapter

from <package>.<feature>.cli import APP_COMMANDS


_adapter = PydanticSettingsTyperAdapter()
app = _adapter.as_typer(
    APP_COMMANDS,
    prog="robo-orchard <group>",
    description="<Package command group help>.",
)
```

## Entry Point

For pyproject-based packages:

```toml
[project.entry-points."robo_orchard.cli"]
<group> = "<package>.cli_plugin:app"
```

For setup.py-based packages:

```python
entry_points={
    "robo_orchard.cli": [
        "<group>=<package>.cli_plugin:app",
    ],
}
```

## Optional Built-In Root Command

Only use this pattern inside `robo_orchard_core.tools.cli` for root-owned
built-ins that should appear even when optional dependencies are missing.

```python
BUILTIN_CLI_EXTENSIONS = {
    "<command>": BuiltinCliExtension(
        target="<module>:cli_app",
        help="<help text>.",
        required_modules=("<optional_module>",),
        requirement="<package>[<extra>]",
    ),
}
```

## Legacy Console Wrapper

Keep this only when an existing console script must remain callable.

```python
import argparse
import sys


DEPRECATED_ENTRY_WARNING = (
    "Warning: <old-entrypoint> is deprecated; use "
    "`robo-orchard <group> <command>` instead."
)


def legacy_cli() -> None:
    """Run the deprecated standalone console entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED: use `robo-orchard <group> <command>` instead.\n\n"
            "<old entrypoint help>."
        )
    )
    if not _is_help_request(sys.argv):
        print(DEPRECATED_ENTRY_WARNING, file=sys.stderr)
    args = pydantic_from_argparse(LegacyCliParam, parser)
    args.command_impl()


def _is_help_request(argv: list[str]) -> bool:
    return any(item in {"--help", "-h"} for item in argv[1:])
```

## Test Checklist

- root help lists the plugin group
- group help lists the leaf commands in stable order
- leaf command invokes the expected `command_impl()`
- pydantic-settings help is visible for leaf options
- kebab-case options work in examples
- snake_case options work only when compatibility is required
- root CLI does not import optional heavy modules during startup
- optional dependency placeholder stays visible and reports the missing extra
- legacy wrapper warns on non-help invocation and does not warn on help
