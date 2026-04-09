---
orphan: true
---

# Architecture Overview

This document provides a high-level architecture overview of
`robo_orchard_core`. It describes the intended layering, the current package
map, and the design constraints that help keep the repository maintainable as
it grows.

## Design Goals

- Keep core robotics data structures and math utilities reusable outside of
  any specific runtime.
- Make environment, policy, controller, and kinematics abstractions stable
  enough to support both simulation and real hardware backends.
- Isolate runtime-specific integrations such as Ray, Jupyter widgets, CLI
  tools, and device adapters so they can evolve without destabilizing core
  APIs.
- Prefer explicit lifecycle and data-shape contracts over implicit behavior.
- Keep configuration declarative, inspectable, and safe to validate.

## Package Map

The current source tree is organized around a few major responsibilities:

- `robo_orchard_core/datatypes`
  - Batched robotics data structures such as transforms, camera data, and
    joint state.
- `robo_orchard_core/utils`
  - Shared helpers for config loading, math, timing, logging, dict/string
    utilities, and Ray integration.
- `robo_orchard_core/envs`
  - Environment abstractions plus the manager-based environment framework.
- `robo_orchard_core/policy`
  - Policy interfaces and remote wrappers.
- `robo_orchard_core/controllers`
  - Control algorithms such as differential IK.
- `robo_orchard_core/kinematics`
  - Chain abstractions and kinematic computation helpers.
- `robo_orchard_core/devices`
  - Device-facing adapters, including camera interfaces.
- `robo_orchard_core/tools`
  - Small operational and developer-facing tools such as the file-server CLI.
- `robo_orchard_core/viz`
  - Visualization helpers, currently focused on notebook-centric workflows.

## Module Responsibility Matrix

The table below is a practical guide for deciding what belongs in each module
area and what should stay out.

| Area | Primary responsibility | Should usually avoid |
| --- | --- | --- |
| `datatypes` | Canonical batched data containers and conversion semantics | Runtime-specific device code, notebook widgets, Ray transport logic |
| `utils/math` | Pure math and geometric operations | Environment state management, device I/O, visualization concerns |
| `utils` | Shared low-level helpers with broad reuse | Domain-specific workflow logic, long-lived runtime orchestration |
| `envs` | Environment contracts and environment-level orchestration | Notebook UI code, CLI handling, raw utility dumping ground |
| `envs/managers` | Reusable term orchestration inside environments | Transport-specific logic, ad-hoc environment hacks outside manager contracts |
| `policy` | Policy interfaces and concrete policy implementations | Environment internals, visualization-specific behavior |
| `controllers` | Control algorithms and command-generation logic | Ray wrappers, CLI code, notebook plumbing |
| `kinematics` | Robot structure and forward/inverse kinematics helpers | Device transport, visualization runtime behavior |
| `devices` | Adapters from external devices into repository contracts | Core math helpers, environment orchestration rules |
| `tools` | Operator and developer-facing entry points | Core business logic that should live in reusable modules |
| `viz` | Visualization adapters on top of normalized data | Core datatypes, hidden lifecycle ownership of domain objects |

The most important rule is to keep ownership narrow. If a module starts doing
two jobs at once, it usually becomes harder to test and reason about.

## Intended Layering

The repository already contains the ingredients for a clean layered design.
The main recommendation is to make that layering explicit and preserve it
consistently.

### 1. Core Layer

This layer should contain pure, runtime-agnostic building blocks:

- `datatypes`
- `utils/math`
- safe and general-purpose parts of `utils`
- low-level config schemas and serializers

Core-layer code should:

- avoid depending on Ray, Jupyter, FastAPI, CLI frameworks, or hardware
  adapters
- keep batch semantics explicit
- fail fast on invalid input
- be easy to test without external services

### 2. Domain Layer

This layer defines robotics concepts and framework contracts:

- `envs`
- `envs/managers`
- `policy`
- `controllers`
- `kinematics`

Domain-layer code should:

- define protocols, interfaces, and lifecycle expectations
- describe how data flows between policies, managers, environments, and
  controllers
- depend on the core layer, but not on notebook, CLI, or service adapters

### 3. Integration Layer

This layer binds the domain model to specific runtimes:

- `devices`
- `tools`
- `viz/jupyter`
- remote wrappers currently implemented around Ray

Some modules under `utils`, especially `utils/ray.py`, conceptually behave as
integration-layer code even though they currently live in a shared utility
package. Even if the package layout stays the same for compatibility, the code
should still be treated as an adapter boundary.

## Main Architectural Contracts

The repository works best when a few cross-cutting contracts stay stable.

### Batch-First Data Contract

Datatypes should make batch semantics explicit and consistent:

- single-instance adapters should upcast to batch size `1`
- concatenation helpers should validate optional metadata carefully
- shape mismatches should fail early with readable errors

This matters most for camera, transform, and observation-related data.

### Explicit Lifecycle Contract

Long-lived runtime objects should expose deterministic cleanup paths:

- environments and remote wrappers should provide `close()`
- executors and visualization widgets should release threads, handles, and
  remote resources explicitly
- destructors should be best-effort only, not the primary shutdown mechanism

### Manager Contract

The manager system is a core orchestration mechanism and should behave like a
framework within the framework. Each manager type should have a clear contract
for:

- construction and validation
- reset/update/apply order
- expected shapes and spaces
- error handling for missing or malformed terms

### Declarative Config Contract

Configuration should describe what to build, not act as a backdoor execution
mechanism. In practice this means:

- prefer structured config objects over arbitrary strings
- prefer explicit registries over dynamic evaluation
- keep config parsing predictable, testable, and safe

## Typical Data Flows

The architecture becomes easier to reason about when the main runtime flows are
described explicitly.

### Config to Runtime Object

The common construction flow is:

1. A typed config object declares what should be constructed.
2. The config resolves the target class or adapter.
3. The resulting object exposes a domain-facing interface such as
   environment, policy, or manager behavior.

The key architectural rule is that config objects should remain declarative.
They may select implementations, but they should avoid embedding arbitrary
runtime logic.

### Manager-Based Environment Loop

The intended manager-based environment flow is:

```text
policy -> action manager -> environment state update -> event manager
      -> observation manager -> observations/rewards/info
```

More concretely:

1. A policy receives observations and produces actions.
2. The action manager validates, preprocesses, and applies those actions.
3. The environment updates its underlying world or device state.
4. The event manager emits lifecycle notifications such as reset and step.
5. The observation manager assembles the next observation payload.
6. The environment returns an `EnvStepReturn`.

This loop is one of the most important framework-level flows in the repository.
If this flow changes, the manager contracts and tests should usually change
with it.

#### Step Sequence Sketch

The following sequence shows the intended interaction shape for one environment
step:

```text
Policy.act(obs)
    -> action payload
ActionManager.process(actions)
    -> validated / normalized action state
ActionManager.apply()
    -> environment-side actuation
Environment.step(...)
    -> update scene or device state
EventManager.notify("step", StepEvent(...))
    -> lifecycle side effects
ObservationManager.get_observations()
    -> next observation payload
Environment returns EnvStepReturn(...)
```

This sequence is intentionally simple:

- the policy decides
- the action manager translates
- the environment mutates state
- the event manager broadcasts lifecycle events
- the observation manager packages the next readout

If code starts bypassing these responsibilities, it is usually a sign that the
abstraction boundary is drifting.

### Remote Wrapper Flow

The remote flow should be understood as an adapter around domain objects, not
as a separate domain model:

```text
config -> remote wrapper -> Ray actor -> concrete env/policy implementation
```

The wrapper should preserve the local contract as much as possible while adding
only:

- remote initialization behavior
- readiness checks
- explicit cleanup
- serialization boundaries

The remote wrapper should not redefine the semantics of the underlying policy
or environment.

#### Remote Lifecycle Sequence Sketch

The expected remote lifecycle can be summarized as:

```text
Config.as_remote(...)
    -> Remote*Config
Remote*Config()
    -> initialize Ray if needed
    -> create remote actor
    -> check readiness
Caller uses remote wrapper
    -> wrapper forwards method calls to actor
Caller calls close()
    -> best-effort remote-side close()
    -> ray.kill(actor)
    -> local handle cleared
```

This sequence is important because remote wrappers are adapters, not owners of
different business logic. Their job is to preserve the local contract while
adding transport and cleanup behavior.

### Device and Visualization Flow

Device and visualization flows should remain downstream of the core/domain
layers:

```text
device adapter -> datatypes -> domain object -> visualization / tool adapter
```

For example, a camera adapter should normalize raw sensor output into the
repository's batched datatypes first. Notebook widgets or CLI tools should then
consume those normalized representations instead of inventing their own shape
or lifecycle rules.

## Recommended Dependency Direction

The preferred dependency direction is:

```text
Core -> Domain -> Integrations
```

More concretely:

- `datatypes` and pure `utils` should not depend on `envs`, `policy`, `viz`,
  or `tools`
- `envs`, `policy`, `controllers`, and `kinematics` can depend on core
  modules, but should not require notebook or CLI code
- `devices`, `viz`, `tools`, and Ray wrappers can depend on both core and
  domain modules

If a new feature needs to cross these boundaries, it is usually a signal that
the shared abstraction should be lifted into the core or domain layer instead
of wiring integration code directly into lower-level modules.

## Placement Heuristics

When deciding where a new file or class should live, these heuristics are
usually enough:

### Put It in `datatypes` If

- it represents a reusable robotics value object
- its main job is validation, batch semantics, or structural conversion
- it should be consumable by environments, devices, and visualization equally

### Put It in `envs` or `envs/managers` If

- it participates directly in reset/step/update orchestration
- it consumes or produces environment observations, actions, rewards, or
  events
- it defines framework behavior for term composition

### Put It in `policy` If

- it maps observations to actions
- it wraps or composes policy behavior
- it should be substitutable anywhere a policy interface is expected

### Put It in `devices` If

- it talks to an external sensor or actuator
- it converts raw hardware output into repository datatypes
- it depends on hardware-side assumptions that should not leak into the core

### Put It in `viz` or `tools` If

- it is primarily for human interaction, debugging, inspection, or operations
- it can be removed without changing the core domain model
- it depends on UI, notebook, CLI, or service-entry concerns

### Put It in `utils` Only If

- it is genuinely cross-cutting
- it is not specific to a single domain area
- a more explicit home such as `datatypes`, `envs`, or `devices` would be
  misleading

`utils` is useful, but it should not become the default destination for code
that merely lacks a better home. If the helper is primarily about one domain,
it is usually better to keep it near that domain.

## Stable and Evolving Interfaces

Not every module in the repository needs the same stability promise. Making
that explicit helps contributors know where to extend the system and where to
avoid tight coupling.

### Interfaces That Should Be Stable

These are the interfaces downstream users are most likely to depend on:

- typed config objects and class-factory patterns
- batched datatypes and their shape conventions
- `EnvBase` and `EnvStepReturn`
- `PolicyMixin`
- manager configuration and manager lifecycle hooks
- explicit `close()` behavior for long-lived runtime objects

For these interfaces, the most important kind of stability is behavioral
stability:

- argument meaning should stay consistent
- batch semantics should stay consistent
- return shapes should stay consistent
- lifecycle methods should remain explicit and predictable

Changes to these areas should usually come with:

- migration notes when behavior changes
- contract tests
- updated documentation

### Interfaces That Are Good Extension Points

These are the places where adding new implementations should be encouraged:

- new config subclasses
- new manager terms and manager types
- new environment or policy implementations
- new device adapters
- new remote or visualization adapters

The goal is to make extension cheap without making the public contract vague.

### Interfaces That Should Be Treated as Internal

The following areas are better treated as internal implementation details until
they are documented more formally:

- private helper behavior in generic utility modules
- ordering or caching internals that are not part of documented contracts
- notebook widget plumbing
- Ray-specific cleanup, readiness, and transport details
- doc-generation internals under `docs/doc_gen`

These can still evolve quickly, but they should not quietly leak into the
public API surface.

## Public API Guidance

As a practical rule, downstream projects should prefer importing from modules
that define framework concepts rather than from low-level helper internals.

Preferred imports usually look like:

- environment and manager abstractions from `envs`
- policy abstractions from `policy`
- data containers from `datatypes`
- documented config and registry helpers from `utils`

Less desirable integration points are:

- undocumented private helpers
- transport-specific behavior hidden inside remote helpers
- notebook-only widget details
- utility implementation details that are not part of a documented contract

This guidance does not need to be perfect on day one, but documenting it helps
the package evolve toward a more intentional API surface.

## Suggested Extension Points

When extending the repository, these are the preferred entry points:

- add new batched values or conversion helpers under `datatypes`
- add new orchestration behavior through managers and environment contracts
- add new runtime backends through adapter-style wrappers
- add user-facing operational features in `tools`
- add notebook-only visualization behavior in `viz/jupyter`

This keeps new code discoverable and limits accidental coupling.

## Current Architectural Debts

The current codebase is already quite capable, but a few recurring issues point
to architectural cleanup opportunities:

- some public APIs still rely on implicit shape or lifecycle assumptions
- parts of the remote/runtime integration layer are still mixed into `utils`
- several important contracts are enforced by implementation details rather
  than dedicated contract tests
- optional dependency boundaries exist in packaging, but are not yet fully
  reflected in module boundaries and test grouping

## Recommended Next Steps

The following steps would strengthen the architecture without requiring a large
rewrite:

1. Add contract tests for camera, manager, and remote-wrapper behavior.
2. Continue moving from string-based dynamic config behavior toward explicit
   registries.
3. Make adapter boundaries more visible in code organization and docs,
   especially around Ray, notebook visualization, and device integrations.
4. Define a short list of stable public interfaces that downstream projects can
   rely on.
5. Group tests and optional dependencies around architectural layers so
   integration-heavy modules are easier to validate in isolation.

## Review Checklist for New Features

When adding a new module or feature, these questions help keep the
architecture aligned:

- Is this a core concept, a domain concept, or an integration concern?
- Does it introduce a new public contract, or implement an existing one?
- Are batch shape and lifecycle semantics explicit?
- If it depends on optional packages, is that dependency isolated cleanly?
- Would a downstream user know where this feature belongs from the package
  layout and docs?

## Non-Goals

This architecture overview does not require an immediate package rename or a
large directory migration. The main goal is to clarify the intended design
direction so incremental changes move the repository toward cleaner boundaries
instead of away from them.
