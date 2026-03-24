# RoboOrchardCore

**RoboOrchardCore** is the core package of the project **RoboOrchard**, which provides the basic infrastructure for the framework, such as configuration management, data structure, environment abstraction, etc.

## What It Includes

- Structured config objects and class factories built on top of `pydantic`
- Batched robotics datatypes for geometry, transforms, cameras, and joint state
- Environment, manager, and policy abstractions for simulation or real-world systems
- Ray-based remote wrappers for environments and policies
- Small developer tools such as the bundled file server CLI

## Installation

Install the package in editable mode:

```bash
make install-editable
```

Install development dependencies for linting, tests, and docs:

```bash
make dev-env
```

If you prefer pip directly, the package also exposes optional extras such as
`[all]`, `[tools]`, and `[virtual_desktop]`.

## Quick Start

Create and round-trip a typed config:

```python
from robo_orchard_core.utils.config import Config


class DemoConfig(Config):
    name: str = "demo"
    steps: int = 4


cfg = DemoConfig()
yaml_text = cfg.to_str(format="yaml")
loaded = DemoConfig.from_str(yaml_text, format="yaml")

print(loaded.to_dict())
```

Start the bundled file server from the CLI:

```bash
robo-orchard file-server --dir .
```

## Development Workflow

- Run lint: `make check-lint`
- Auto-format: `make auto-format`
- Type check: `pyright`
- Build docs: `make doc`
- Run the full Python test suite: `make test`

Test reports are written under `build/test/`.

## Test Prerequisites

- Full asset-backed tests expect `ROBO_ORCHARD_TEST_WORKSPACE` to point to a
  workspace that contains the required robot and image assets.
- Remote environment and policy tests use Ray; ensure Ray is installed and
  available in the active environment.
- If Ray is not already running in your setup, you can usually start a local
  instance with `ray start --head`.

## Documentation

The source documentation lives under [`docs/`](docs/).
Run `make dev-env` before `make doc` so the required Sphinx dependencies are
available.
