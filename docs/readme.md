# RoboOrchardCore Documentation

RoboOrchardCore is the shared runtime layer behind the RoboOrchard package
stack. This documentation site is the best entry point when you want to learn
the package model, understand the system structure, and browse the generated
API surface in one place. For repository-oriented information such as badges,
package positioning, and source navigation, see the
[repository README](https://github.com/HorizonRobotics/RoboOrchardCore#readme).

> Use this site when you want guided docs and concept-level navigation. Use
> the repository README when you want install context, source layout, or
> contribution entry points.

## Navigate By Task

| Goal | Start here | What you will get |
| --- | --- | --- |
| Install the package correctly | {doc}`Installation guide <build/gallery/get_started/installation/index>` | Setup instructions for the supported install paths and extras. |
| Understand the package from whole to part | {doc}`System Overview tutorials <build/gallery/tutorials/system_overview/index>` | A narrative overview of config design, runtime flow, remote execution, control layers, and tools. |
| Learn the core tensor-backed containers | {doc}`Data Types tutorials <build/gallery/tutorials/datatypes/index>` | Practical introductions to transforms, camera data, and related structures. |
| Browse symbol-level APIs | {doc}`API reference <autoapi/index>` | Generated module, class, and function documentation. |

## What This Package Covers

- **Typed configuration**: declarative config models and class factories built
  on top of `pydantic`.
- **Batch-first robotics datatypes**: geometry, frame transforms, camera data,
  and joint state containers designed for explicit tensor shape handling.
- **Manager-driven runtime abstractions**: environment, policy, controller,
  and remote execution building blocks for simulation or real-world systems.
- **Tools and integrations**: CLI helpers, notebook-oriented visualization,
  and runtime adapters around concrete integration boundaries.

## A Simple Mental Model

Think about RoboOrchardCore in four parts:

- **Config says what to build**: typed configs and class factories keep runtime construction explicit.
- **Datatypes say how data moves**: batch-first transforms, camera payloads, and joint state define shared contracts.
- **Runtime modules say who does what**: environments, policies, controllers, and remote wrappers coordinate decisions and execution.
- **Tools and integrations say how systems enter**: CLI commands, device adapters, and visualization layers expose the runtime model outward.

## System Overview Topics

- {doc}`Best Practices for Config <build/gallery/tutorials/system_overview/nonb-01_config_design>` explains how typed configs stay declarative and validation-friendly.
- {doc}`Environment and Managers <build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` describes the local runtime loop and manager boundaries.
- {doc}`Remote Execution with Ray <build/gallery/tutorials/system_overview/nonb-03_remote_execution>` shows how local configs scale to actor-backed execution.
- {doc}`Controllers, Kinematics, and Motion Commands <build/gallery/tutorials/system_overview/nonb-04_control_and_kinematics>` explains how motion goals flow toward executable commands.
- {doc}`CLI Tools and Extensions <build/gallery/tutorials/system_overview/nonb-05_tools_and_extensions>` covers operator-facing entry points and plugin packaging.

## Recommended Reading Paths

### For Package Users

- Start with the
  {doc}`installation guide <build/gallery/get_started/installation/index>`.
- Move to the
  {doc}`System Overview tutorials <build/gallery/tutorials/system_overview/index>`
  when you want the package-wide mental model first.
- Continue with the
  {doc}`Data Types tutorials <build/gallery/tutorials/datatypes/index>` to
  learn the core data model in more detail.

### For Contributors

- Read the
  {doc}`System Overview tutorials <build/gallery/tutorials/system_overview/index>`
  first to understand package boundaries, runtime flow, and module ownership.
- Use the {doc}`API reference <autoapi/index>` when tracing concrete symbols
  and call surfaces.
- See
  [CONTRIBUTING.md](https://github.com/HorizonRobotics/RoboOrchardCore/blob/master/CONTRIBUTING.md)
  for the contribution workflow.

## Key Entry Points

| Start from | Use it when... |
| --- | --- |
| `utils.config.Config` | You need typed runtime configs with predictable serialization. |
| `datatypes.BatchTransform3D` and related datatypes | You need reusable geometry, camera, or joint-state containers. |
| `envs.TermManagerBasedEnv` | You want a manager-driven environment loop with explicit event flow. |
| `policy.RemotePolicy` | You need Ray-backed remote policy execution behind a typed interface. |
| `tools.cli` | You want CLI and service-style entry points from the same package. |

## Documentation Map

| Area | What you will find |
| --- | --- |
| System Overview | Package-wide concepts, runtime flow, and a module-level responsibility map |
| Tutorials | Guided introductions to installation and focused subsystem areas such as datatypes |
| API reference | Auto-generated symbol-level documentation for modules, classes, and functions |

## For Contributors

- Install development dependencies first with `make dev-env`.
- Build the docs with `make doc`.
- See
  [CONTRIBUTING.md](https://github.com/HorizonRobotics/RoboOrchardCore/blob/master/CONTRIBUTING.md)
  for the contribution workflow.
