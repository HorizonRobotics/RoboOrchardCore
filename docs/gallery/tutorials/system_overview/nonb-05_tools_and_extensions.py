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
# application. It currently includes a built-in ``file-server`` command and a
# plugin loading path based on Python entry points.
#
# The built-in file server is a useful example of the intended layering:
#
# - the CLI command parses command-line options such as ``--host`` and
#   ``--port``
# - the service implementation lives in ``tools.simple_file_server``
# - the reusable behavior stays in normal Python modules rather than being
#   embedded directly in the command definition


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
# The top-level CLI scans the ``robo_orchard.plugins`` entry point group and
# mounts discovered Typer applications as subcommands. This means downstream
# packages can extend the command surface without patching the core package.
#
# .. code-block:: toml
#
#    [project.entry-points."robo_orchard.plugins"]
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
# - service modules should validate paths, inputs, and runtime boundaries
#   explicitly
# - plugin commands should reuse existing configs and domain interfaces rather
#   than introducing parallel logic paths
# - optional dependencies should stay isolated to the tools that need them
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
