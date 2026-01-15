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

from importlib.metadata import entry_points

import typer

app = typer.Typer(
    help="Robo Orchard Core CLI - A unified toolset for robotics development.",
    add_completion=False
)


@app.callback()
def main_callback():
    """Robo Orchard Core CLI - A unified toolset for robotics development.

    Use 'robo-orchard [COMMAND] --help' for more information.
    """
    pass


def load_plugins(cli_app: typer.Typer):
    """Dynamically loads external plugins registered via entry points.

    This function scans for the group 'robo_orchard.plugins' and mounts
    discovered Typer apps as sub-commands.

    Args:
        cli_app (typer.Typer): The main CLI application instance.
    """
    plugin_group = "robo_orchard.plugins"

    discovered_plugins = entry_points(group=plugin_group)

    for entry_point in discovered_plugins:
        try:
            plugin_app = entry_point.load()
            cli_app.add_typer(plugin_app, name=entry_point.name)
        except Exception as e:
            print(f"Warning: Failed to load plugin '{entry_point.name}': {e}")


@app.command(name="file-server")
def file_server(
    port: int | None = typer.Option(None, "--port", help="Port to bind the server to."),  # noqa: E501
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind to."),  # noqa: E501
    directory: str = typer.Option(".", "--dir", help="Directory to serve.")
):
    """Starts the simple HTTP file server."""

    from robo_orchard_core.tools.simple_file_server import start_server
    from robo_orchard_core.utils.network import find_free_port

    if port is None:
        port = find_free_port()

    start_server(host=host, port=port, directory=directory)


load_plugins(app)


def main():
    app()


if __name__ == "__main__":
    main()
