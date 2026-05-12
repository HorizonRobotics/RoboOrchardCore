# CLI Guideline

Use this reference when adding, migrating, or reviewing `robo-orchard` CLI
commands, including the root CLI, plugin command groups, optional built-ins,
legacy wrappers, and Typer/pydantic-settings leaf-command integration.

## CLI Ownership Model

- Let Typer own command trees, plugin loading, command grouping, and the root
  help shell.
- Let pydantic-settings own leaf-command parameter parsing, validation,
  defaults, config aliases, and `command_impl()` execution.
- Prefer `SettingConfig` leaf classes for new executable commands. Avoid new
  `CliSubCommand` aggregate classes in canonical command paths.
- Use `PydanticSettingsTyperAdapter.as_typer(...)` as the bridge:
  - pass one executable `SettingConfig` subclass for a single leaf app
  - pass `Mapping[str, SettingConfigClass]` for an explicit command group
- Keep command maps close to the feature module that owns the command models.
  The root CLI should mount apps, not import feature implementation details.

## CLI Bridge Adapter Contract

- Adapted settings classes must be `BaseSettings` / `SettingConfig`
  subclasses and expose `command_impl()`.
- The adapter intentionally rejects `CliSubCommand` aggregate settings
  classes. Split those aggregates into a command-name mapping instead.
- The adapter owns the leaf Typer callback and automatically sets the Typer
  forwarding behavior needed by pydantic-settings help and parsing.
- Adapter-generated leaves should parse only the current leaf's `ctx.args`;
  they must not re-read global `sys.argv`.
- Adapter-generated pydantic-settings leaves own `context_settings` with
  `allow_extra_args` / `ignore_unknown_options` and `add_help_option=False`.
  Root apps, plugin groups, and ordinary Typer commands should keep Typer's
  default argument checking so misspelled command-tree options are not hidden.
- Prefer kebab-case in public examples. Keep snake_case support only as
  compatibility at the adapter boundary.
- If a leaf returns a value from `command_impl()`, treat it as
  command-specific. The adapter does not interpret it.

## Plugin And Optional Dependency Loading

- Register package-level CLI apps through the `robo_orchard.cli` entry point
  group.
- Built-in root commands that depend on optional extras should be loaded by
  spec string after dependency checks, not imported directly by
  `robo_orchard_core.tools.cli`.
- When optional dependencies are missing, keep a placeholder command mounted
  so root help remains stable and command invocation reports the missing
  requirement.
- If an external entry point duplicates a built-in command name, the built-in
  command owns that name. Warn and skip the duplicate, identifying both the
  already-loaded spec and the skipped spec.

## Legacy Entry Points

- Keep legacy console scripts only when compatibility is deliberate.
- Legacy wrappers should delegate to the existing settings model or command
  implementation, and should not become a second canonical execution path.
- Non-help legacy invocations should warn on stderr and point to the unified
  `robo-orchard ...` command.
- Help invocations may mark the command as deprecated in help text but should
  not emit an extra stderr warning.

## Test Expectations

- Cover root help and group help separately.
- Cover that a leaf command reaches its `command_impl()`.
- Cover both kebab-case public options and legacy snake_case options when
  compatibility is expected.
- Cover optional-dependency placeholders without importing the heavy module.
- Cover that root CLI startup does not import feature modules that are meant
  to be lazy.
- Cover legacy wrapper warnings when keeping old console scripts.
- For negative tests that intentionally pass the wrong static type into the
  adapter, cast the value to `Any` so pyright does not reject the test before
  runtime validation is exercised.

## Anti-Patterns

- Do not make the root CLI import an optional feature module only to register
  the command.
- Do not keep both a `CliSubCommand` aggregate and a Typer command map as
  competing public command trees.
- Do not add a custom Typer callback for every settings leaf unless the
  adapter cannot express the behavior.
- Do not preserve old backend selection switches in the canonical path after a
  backend has been intentionally removed.
