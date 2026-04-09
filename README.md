# RoboOrchardCore

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Platform](https://img.shields.io/badge/platform-linux--64-green.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-source-brightgreen.svg)](docs/)

**RoboOrchardCore** is the runtime substrate for **RoboOrchard** packages. It
gives robotics teams a shared foundation for typed configuration,
tensor-backed datatypes, manager-driven environment loops, and operator-facing
tooling so simulation, training, and deployment code can share one contract.

> Use RoboOrchardCore when you need a reusable robotics runtime substrate that
> keeps data shapes explicit, configuration type-safe, and integration
> boundaries clean across simulation and real-world systems.

[Docs Home](docs/readme.md) | [System Overview](docs/gallery/tutorials/system_overview) | [Contributing](CONTRIBUTING.md)

## Recommended Learning Path

If you are new to RoboOrchardCore, use this as the default reading path:

1. Install the package with the path that matches your workflow.
2. Read the [system overview tutorials](docs/gallery/tutorials/system_overview) to build the package-wide mental model.
3. Continue to the [data types tutorials](docs/gallery/tutorials/datatypes) when you want deeper detail on the shared containers.
4. Finish with the [API reference](docs/autoapi) when you need concrete class and module-level details.

## Start Here By Workflow

| If you want to... | Start with... | Why it is the right entry point |
| --- | --- | --- |
| Build typed runtime configs and factories | `robo_orchard_core.utils.config.Config` and `ClassConfig` | They keep construction declarative, serializable, and validation-friendly. |
| Normalize robot state, transforms, cameras, or joints | `robo_orchard_core.datatypes` | The package ships batch-first containers that keep tensor shape semantics explicit. |
| Assemble manager-driven environment loops | `robo_orchard_core.envs.TermManagerBasedEnv` | It wires action, observation, and event managers into one reusable control loop. |
| Run policies out of process with Ray | `robo_orchard_core.policy.RemotePolicy` | It wraps policy execution behind a typed remote boundary instead of ad-hoc actor code. |
| Ship operational commands or services | `robo_orchard_core.tools.cli` | The built-in Typer CLI exposes commands today and can load plugins through entry points. |

## Why Teams Use It

- **Configs stay declarative**: `Config` and `ClassConfig` support structured
  construction, validation, and round-trip serialization for YAML, TOML, and
  JSON workflows.
- **Data contracts stay explicit**: tensor-backed datatypes for transforms,
  camera payloads, frame graphs, and joint state keep batch semantics stable
  across simulation and deployment code.
- **Runtime boundaries stay clean**: environments, policies, controllers, and
  remote wrappers are separated from notebook, CLI, and device adapters.
- **Installation stays modular**: install only the extras you need for
  kinematics, notebook visualization, virtual desktop workflows, or tools.

## Package Surface

| Area | What it gives you | Representative entry points |
| --- | --- | --- |
| Configuration | Typed configs, class factories, and serialization helpers | `utils.config.Config`, `utils.config.ClassConfig` |
| Datatypes | Shared batched containers for geometry, cameras, joints, and frame graphs | `datatypes.BatchTransform3D`, `BatchFrameTransform`, `BatchCameraData`, `BatchJointsState` |
| Environments | Manager-driven action, observation, and event orchestration | `envs.TermManagerBasedEnv`, `envs.managers` |
| Policies and remoting | Local policy contracts plus Ray-backed remote execution | `policy.base`, `policy.RemotePolicy` |
| Control and kinematics | Robot control primitives and kinematics helpers | `controllers`, `kinematics` |
| Tools and visualization | CLI commands, service helpers, notebook-centric visualization | `tools.cli`, `viz.jupyter`, `devices` |

For a guided whole-to-part walkthrough of package responsibilities and module
areas, see the
[system overview tutorials](docs/gallery/tutorials/system_overview),
which cover config design, manager-driven runtimes, Ray execution,
control/kinematics boundaries, and operator-facing tools.

## Installation

Choose the install path that matches your use case. For official published
releases, prefer installing the package directly from PyPI.

| Use case | Command | When to use |
| --- | --- | --- |
| Official released package | `pip install robo_orchard_core` | Choose this when you want the latest published release without cloning the repository. |
| Base package from source | `pip install .` | Choose this when you only need the core package without optional extras. |
| Full feature set | `make install` or `pip install ".[all]"` | Choose this when you want the default source install path with the full extra set enabled. |
| Editable development setup | `make install-editable` | Choose this when you plan to modify the package locally and want an editable install. |
| Development dependencies | `make dev-env` | After either install path above, run this when you need lint, test, and docs tooling plus pre-commit hooks. |

Selected extras are also available for narrower setups:

| Extra | Installs support for... | Use it when... |
| --- | --- | --- |
| `tools` | Typer, FastAPI, aiofiles, and uvicorn-backed tooling | You need the bundled CLI or service-style helpers. |
| `kinematic` | `pytorch_kinematics` integration | You need chain or kinematics-oriented functionality. |
| `ipy_viz` | Notebook visualization widgets and canvas stack | You want interactive Jupyter visualization. |
| `virtual_desktop` | Notebook widgets plus desktop automation dependencies | You need remote desktop or GUI-oriented notebook workflows. |

After installation, use the documentation paths that match your goal:

- Start with [docs/readme.md](docs/readme.md) for the overall reading map.
- Read [docs/gallery/tutorials/system_overview](docs/gallery/tutorials/system_overview) when you want a package-wide walkthrough from overall structure to local modules, including remote execution, control, and tools.
- Use [docs/](docs/) when you want tutorials, API docs, and contribution-oriented source material.

## Scope And Constraints

- The documented default environment is Ubuntu 22.04 with Python 3.10.
- Several capabilities live behind optional extras such as `kinematic`, `tools`, `ipy_viz`, and `virtual_desktop`.
- Distributed execution and some higher-level workflows assume Ray is installed and available.
- If you only need a few standalone math helpers or a one-off CLI, this package may be broader than necessary; it is most useful when you want one shared runtime model across config, data contracts, environments, policies, and tools.

## Typical Use Cases

- Building typed robotics application configs that can be serialized to YAML,
  TOML, or JSON and loaded back with validation.
- Standardizing transforms, camera frames, and joint state using shared batched
  datatypes across simulation and deployment code.
- Implementing manager-driven environments with explicit action,
  observation, and event contracts.
- Wrapping policy execution behind a Ray-backed remote interface instead of
  embedding actor management into model code.
- Shipping small operational utilities from the same package ecosystem, such
  as the bundled file server CLI.

## Development Workflow

Run `make dev-env` first so the development-only tools are available in your
active environment.

- Run lint: `make check-lint`
- Auto-format: `make auto-format`
- Type check: `pyright`
- Build docs: `make doc`
- Run the full Python test suite: `make test`

Test reports are written under `build/test/`.

## Before Running Tests

- Integration tests that depend on robot models or image assets expect
  `ROBO_ORCHARD_TEST_WORKSPACE` to point to a workspace containing the required
  test assets.
- Distributed environment and policy tests use Ray; ensure Ray is installed
  and available in the active environment.
- If Ray is not already running in your setup, you can usually start a local
  instance with `ray start --head`.

## Documentation And Further Reading

- Source docs:
  [docs/](docs/)
- System overview source:
  [docs/gallery/tutorials/system_overview](docs/gallery/tutorials/system_overview)
- Docs landing page source:
  [docs/readme.md](docs/readme.md)
- Contribution workflow:
  [CONTRIBUTING.md](CONTRIBUTING.md)

Run `make doc` after `make dev-env` to build the local HTML docs.

## License

**RoboOrchardCore** is licensed under the
[Apache License 2.0](LICENSE).
Contributions are welcome through the standard repository workflow.
