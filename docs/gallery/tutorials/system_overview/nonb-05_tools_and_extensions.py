# ruff: noqa: E501 D415 D205

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

"""CLI Tools and Extensions
========================================
"""

# %%
# Why Tools Live In The Core Package
# ----------------------------------
#
# ``robo_orchard_core`` is not only a library of data structures and runtime
# abstractions. It also ships operator-facing entry points for lightweight
# services and utility workflows. These tools belong in the core package when
# they are reusable across multiple higher-level applications and can be built
# on the same shared runtime contracts.
#
# The guiding rule is simple: keep the runtime logic reusable, and keep the CLI
# or service layer thin.


# %%
# When To Read This Tutorial
# --------------------------
#
# Read this tutorial when the package model already makes sense, and the next
# question is how to expose that model to developers, operators, or service
# workflows without hard-coding everything into one application entry point.


# %%
# The Built-In CLI Structure
# --------------------------
#
# ``robo_orchard_core.tools.cli`` defines the top-level ``robo-orchard`` Typer
# application. It owns the root command tree, root help shell, built-in command
# registration, and package extension discovery. It does not own the runtime
# logic behind each command.
#
# The intended CLI architecture is:
#
# .. code-block:: text
#
#    robo-orchard console script
#      -> root Typer app from robo_orchard_core.tools.cli
#        -> built-in commands and robo_orchard.cli entry points
#          -> package-owned Typer extension app
#            -> Typer command groups and leaf commands
#              -> optional pydantic-settings leaf model
#                -> command_impl()
#
# This split is deliberate. Typer is a good fit for command trees, command
# grouping, plugin loading, and top-level help. ``pydantic-settings`` is a good
# fit for complex leaf-command parameters, environment/config-source merging,
# validation, and executable settings models. The root CLI should provide one
# coherent command surface without becoming the source of truth for every
# command's parameter schema.
#
# The built-in file server is a useful example of the intended layering:
#
# - the root CLI mounts ``file-server`` by an entry-point-style spec instead of
#   importing the implementation directly
# - the command's settings-backed leaf lives in ``tools.simple_file_server``
# - the reusable behavior stays in normal Python modules rather than being
#   embedded directly in the command definition
# - missing optional web dependencies produce a stable placeholder command
#   instead of removing ``file-server`` from root help


# %%
# Minimal Plugin App Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The following snippet shows the smallest practical Typer app that can be
# mounted by the top-level plugin discovery path.

import typer

plugin_app = typer.Typer(help="Demo plugin app")


@plugin_app.command()
def status(verbose: bool = False):
    print({"verbose": verbose})


print(plugin_app.info.help)
print(len(plugin_app.registered_commands))
status(verbose=True)


# %%
# Extending The CLI With Plugins
# ------------------------------
#
# The canonical extension point is ``robo_orchard.cli``. The top-level CLI scans
# that entry point group and mounts discovered Typer applications as
# subcommands. This means downstream packages can extend the command surface
# without patching the core package.
#
# .. code-block:: toml
#
#    [project.entry-points."robo_orchard.cli"]
#    my-tool = "my_package.cli:app"
#
# .. code-block:: python
#
#    import typer
#
#    app = typer.Typer()
#
#    @app.command()
#    def status():
#        print("ready")
#
# The architectural benefit is that packaging, discovery, and command routing
# happen in one place while business logic can remain in the downstream module
# that owns it.
#
# The entry point value must load to a ``typer.Typer`` instance. The entry
# point name becomes the command mounted under ``robo-orchard``. For example,
# the entry point above exposes:
#
# .. code-block:: text
#
#    robo-orchard my-tool status
#
# The older ``robo_orchard.plugins`` entry point group may still be loaded as a
# compatibility path, but new packages should use ``robo_orchard.cli``. Built-in
# command names and earlier-loaded extension names win over later duplicates;
# duplicate-name warnings report both the spec that stays loaded and the spec
# that was skipped. Failed extensions warn instead of silently changing the
# command tree.


# %%
# Settings-Backed Leaf Commands
# -----------------------------
#
# Simple commands can use ordinary Typer callbacks. More complex leaf commands
# should usually keep their parameter model in ``pydantic-settings`` and use
# ``robo_orchard_core.tools.cli_bridge.PydanticSettingsTyperAdapter`` to mount
# that model under a Typer command path.
#
# .. code-block:: python
#
#    import typer
#    from pydantic import Field
#
#    from robo_orchard_core.tools.cli_bridge import (
#        PydanticSettingsTyperAdapter,
#    )
#    from robo_orchard_core.utils.cli import SettingConfig
#
#
#    class RunConfig(SettingConfig):
#        """Run the tool."""
#
#        config: str = Field(description="Path to the run config.")
#        dry_run: bool = Field(
#            default=False,
#            description="Preview the operation without executing it.",
#        )
#
#        def command_impl(self) -> None:
#            ...
#
#
#    app = typer.Typer(help="My tool commands.")
#    adapter = PydanticSettingsTyperAdapter()
#    app.add_typer(
#        adapter.as_typer(
#            RunConfig,
#            prog="robo-orchard my-tool run",
#            description="Run the tool.",
#        ),
#        name="run",
#    )
#
# The adapter boundary is intentionally small:
#
# - Typer owns command groups, command names, and root/group help.
# - ``pydantic-settings`` owns leaf parameters, environment/config sources,
#   validation, and ``command_impl()``.
# - The adapter forwards only the current leaf's raw arguments to the settings
#   model; it does not re-read global ``sys.argv``.
# - The adapter does not generate a Typer option signature or expand nested
#   settings fields. Leaf parameters remain a pydantic-settings contract.
# - ``CliSubCommand`` aggregate settings classes are for legacy console scripts,
#   not new ``robo-orchard`` command trees.
#
# For multiple settings-backed leaves, pass an explicit command mapping. The
# mapping is the public command tree at that level. When the mapping app is the
# package entry-point app, expose it directly so ``robo-orchard my-tool run`` is
# not accidentally nested under another ``my-tool`` group:
#
# .. code-block:: python
#
#    class InspectConfig(SettingConfig):
#        """Inspect the tool state."""
#
#        target: str = Field(description="Target to inspect.")
#
#        def command_impl(self) -> None:
#            ...
#
#
#    app = adapter.as_typer(
#        {
#            "run": RunConfig,
#            "inspect": InspectConfig,
#        },
#        prog="robo-orchard my-tool",
#        description="My tool commands.",
#    )


# %%
# Help And Parsing Boundaries
# ---------------------------
#
# The CLI has two help systems at different layers:
#
# .. code-block:: text
#
#    robo-orchard --help
#      # Typer help for root commands and extension groups
#
#    robo-orchard my-tool --help
#      # Typer help for the package-owned command group
#
#    robo-orchard my-tool run --help
#      # pydantic-settings help for the RunConfig leaf parameters
#
# This is the core trade-off of the bridge. Typer remains the command-tree
# source of truth, while pydantic-settings remains the leaf-parameter source of
# truth. The generated leaf uses permissive Typer context settings only at that
# leaf so ``--help`` and validation flags can reach pydantic-settings. Root apps
# and intermediate groups should keep Typer's normal argument checking so
# misspelled command names or group options are not hidden.


# %%
# Deciding Where A Feature Belongs
# --------------------------------
#
# Use ``tools`` when a feature is primarily:
#
# - an operator or developer entry point
# - a small service or utility with a stable command surface
# - reusable across multiple downstream applications
#
# Use ``devices`` when the code is primarily about hardware-facing adapters.
# Use ``viz`` when the code is primarily about presentation or notebook-based
# interaction. Use ``envs`` or other domain modules when the code defines core
# runtime behavior rather than an entry point.


# %%
# Good Extension Hygiene
# ----------------------
#
# A healthy tools layer follows a few rules:
#
# - command functions should stay small and mostly wire arguments into reusable
#   module functions
# - plugin modules and imported settings modules should stay import-light so
#   ``robo-orchard --help`` does not initialize remote SDKs, read credentials,
#   access services, or perform expensive work
# - service modules should validate paths, inputs, and runtime boundaries
#   explicitly
# - plugin commands should reuse existing settings models, configs, and domain
#   interfaces rather than introducing parallel logic paths
# - optional dependencies should stay isolated to the tools that need them
# - complex commands should prefer settings-backed leaves; simple commands can
#   continue to use Typer-native callbacks
#
# This keeps operator-facing workflows easy to add without turning the CLI into
# a second application framework.


# %%
# Relationship To The Rest Of The System
# --------------------------------------
#
# The tools layer sits at the outer edge of the package architecture:
#
# - configs still describe what to construct
# - core datatypes and runtime contracts still define how data moves
# - tools provide the human- or process-facing surface that triggers those
#   reusable internals
#
# This makes tools a good system-overview topic: they show how the package is
# used operationally, not only how it is composed internally.


# %%
# Where To Continue
# -----------------
#
# - Return to :doc:`Remote Execution with Ray </build/gallery/tutorials/system_overview/nonb-03_remote_execution>` when a tool needs to launch or coordinate remote runtime objects.
# - Use :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` if the missing piece is still the core runtime lifecycle rather than the operator-facing shell.
# - Use the :doc:`API reference </autoapi/index>` for concrete CLI and tools modules after the extension pattern is clear.
