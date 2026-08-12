---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
---

# Python Change Instructions

## Core Expectations

- Keep changes compatible with the project's Python version and public APIs unless the task allows otherwise.
- Reuse existing patterns, helpers, constants, and types before adding new ones.
- Keep new logic focused; avoid abstraction added only for style.
- Do not silently swallow exceptions. If catching one, keep enough context to debug it.
- For reusable wrappers around subprocesses, file handles, sockets, or other
  OS-backed resources, keep `open()` / `write()` / `close()` /
  context-manager semantics coherent. Entering a context should represent an
  active resource session, not a closed object that fails only later.
- After startup, write, or finalize failures in an OS-resource wrapper, return
  the object to a stable and diagnosable state before raising. Best-effort
  cleanup should reap child processes or handles when possible and preserve
  cleanup-failure details in the surfaced error.
- When forgetting `close()` is a plausible caller misuse path for a long-lived
  OS-resource wrapper, prefer a best-effort warning or finalizer cleanup path
  in addition to the explicit lifecycle API. Treat that fallback as a safety
  net, not the primary contract.
- Prefer explicit exceptions over `assert` for production-facing runtime
  contracts such as config values, user inputs, and environment settings.

## Module Organization

- When reorganizing a Python module and no stronger local convention exists,
  prefer a public-first layout: module constants/config, core public types,
  public entrypoints, public adapters or mixins, then private helpers.
- When adding a Python module, first decide whether the module itself is an
  intended public interface surface or an internal implementation module. If
  the whole module is internal, prefer an underscore-prefixed filename such as
  `_runner.py` over mechanically prefixing every class and function with `_`.
  Inside an internal module, still distinguish implementation types shared by
  nearby internal modules from module-local helpers; use symbol-level `_` only
  when it adds useful locality or privacy signal.
- A private module filename is not a shortcut around public-surface
  discipline. Package roots, `__all__`, docs, and repository-owned imports
  still determine which paths are supported public APIs.
- Keep canonical types and code paths before legacy or compatibility ones. If
  a legacy type belongs to the same public type family, keep it near that
  family but after the canonical type it complements.
- Keep private persistence, validation, conversion, and compatibility helpers
  after public API definitions unless a local file pattern or readability need
  clearly justifies an earlier placement.
- Before final validation for a new Python module or a deliberate
  export-surface change, define or update a curated `__all__` when the module
  needs to declare supported symbols. Do not add `__all__` to a mature public
  module as incidental cleanup: first audit wildcard-import compatibility in a
  dedicated export-surface change.

## Helper Granularity

- Prefer helpers that represent a real semantic boundary: validation,
  conversion, lifecycle, scheduling, resource cleanup, error handling, or
  reusable domain behavior.
- Inline single-call helpers that only rename, forward, or wrap one operation
  unless they make a non-obvious boundary explicit.
- Avoid splitting one local flow into many tiny helpers when the helper names
  only restate the next line of code.
- Keep helpers near the narrowest layer that owns the behavior. Do not promote
  local policy into shared helpers only because two call sites look similar.
- For bool, enum, or literal parameters that change control flow, resource
  state, threading, or failure behavior, document each meaningful branch in
  the helper docstring.
- Prefer state names that include the owned object or state target, such as
  `evaluator_configured`, `env_prepared`, or `worker_replaced`, instead of
  generic names such as `configured`, `ready`, or `valid`.

## Logging

- When editing an existing file, keep the established nearby logger surface
  unless the task explicitly includes logging cleanup or refactoring.
- Do not rewrite existing modules from one logger helper to another as
  incidental cleanup.
- Keep framework-native loggers when the surrounding module family already
  depends on framework-specific logging behavior.
- For new modules, follow the local logging helper or closest module family
  convention instead of introducing a new logging surface.

## Package Export Surfaces

- Treat repository-owned package `__init__.py` files and their `__all__`
  lists as curated public API surfaces, not as mirrors of every submodule
  symbol.
- Prefer exporting only the most common intended entrypoints from package
  roots. Keep compatibility-only, adapter, resolver, and type-alias symbols in
  their defining submodules unless the package root is intentionally the
  supported import path.
- When adding a new submodule family, default the package root to a minimal
  high-level entrypoint surface. Import tests and internal callers from the
  defining submodule for constants, schema helpers, low-level handles,
  streams, validators, and other implementation details.
- When an old package-level import path must remain for compatibility, prefer
  a deprecated compatibility re-export and update repository-owned imports to
  the defining submodule instead of growing the root surface further.
- If preserving package-root `import *` or `__all__` parity is an explicit
  compatibility requirement during a migration, document that exception in
  package-specific guidance and keep repository-owned imports on the defining
  submodule instead of treating the compatibility export as the preferred new
  entrypoint.
- Before replacing `__getattr__` lazy exports or other dynamic import shims
  for IDE friendliness, identify why the dynamic import exists, such as
  optional dependencies, import cost, circular imports, or registration side
  effects. Preserve the lazy path when that purpose is still valid, and add
  `TYPE_CHECKING` imports for static visibility when helpful.

## Spatial Transform And Matrix Naming

- Follow `.agents/references/spatial-transform-and-matrix-naming-guideline.md`
  for repository-owned pose, transform, and spatial matrix naming. Keep only
  short local mapping comments near code that bridges external conventions to
  repository-owned names.

## Typing

- Preserve or add type annotations when touching function signatures or return values.
- Prefer complete type hints for public APIs, key helpers, and newly added functions unless a clear local pattern or technical reason suggests otherwise.
- For persisted, deserialized, or schema-contract records that need runtime
  validation, prefer Pydantic models.
- For internal runtime structures, builder intermediates, and context objects
  that do not need dict compatibility, prefer `dataclass(slots=True)`.
- For fixed value sets, prefer `Enum` over scattering many string
  `Literal[...]` annotations. Keep explanatory comments next to enum members
  when their meaning is not obvious.
- Use `TypedDict` when the value really remains a plain dict and only static
  typing is needed. Do not use it as the primary representation for contracts
  that need runtime validation.
- When a Pydantic config field is meant to accept a config base class and its
  subclasses, annotate it with `ConfigInstanceOf[BaseConfig]` instead of the
  raw base config type.
- Do not rely on a raw base-config annotation such as `BaseConfig` or
  `BaseConfig[Any]` to express "this base config or any subclass config":
  Pydantic may then serialize only the base fields or emit subtype
  serialization warnings when a subclass config instance is provided.
- For config `class_type` fields that must serialize and deserialize through a
  repository config JSON path, annotate them with `ClassType[T]` instead of a
  bare `type[T]`.
- When validation normalizes a field into a narrower stable runtime type,
  annotate the field with that stored invariant rather than the full raw input
  surface. If callers still need to provide a wider input type, keep that
  wider shape at the validation boundary and expose a conversion helper or
  separate input alias for runtime use.
- Do not leave field annotations implying that normalized-away input types
  still remain available after validation.
- Treat new pyright failures as potential runtime or contract bugs before
  adding type-only workarounds. Prefer proving the runtime shape, narrowing
  with explicit guards, protocols, validators, or helper boundaries, and
  adding focused regression tests for suspicious dynamic data paths.
- Do not use `cast(...)` merely to silence a pyright error. Use it only when
  the runtime invariant has already been established nearby and cannot be
  expressed through a clearer typed API.
- When a typing fix changes import timing for generated code, plugin-loaded
  code, protobuf schemas, or other registration-heavy modules, validate the
  runtime import path. Static correctness is not enough for modules with
  global registries or descriptor pools.

## Documentation and Comments

- Follow the style of nearby code, including imports, naming, file layout, and docstring conventions.
- Add comments only when they provide non-obvious context.
- For local control-flow or data-shape decisions that are easy to misread, add a short adjacent comment at the decision point rather than relying only on a function-level docstring.
- For coordinate-frame transforms, matrix transforms, inversions, or
  convention-bridging code, add a short adjacent comment when the direction
  of the transform or the frame handoff is not obvious from the code alone.
- For public APIs, boundary helpers, or non-obvious interfaces that accept
  or return TF-like structures such as `BatchFrameTransform`,
  `BatchFrameTransformGraph`, camera-pose wrappers, or other frame-bearing
  pose containers, document the frame contract explicitly in the docstring.
- For those TF-like interface values, state the relevant
  `parent_frame_id` and `child_frame_id`. For graph-like values, document
  only the root frame and the specific edge, path, or static edge contract
  that callers rely on.
- Prefer code-near comments for one-off shape normalization, batch unwrapping, side-channel filtering, compatibility branches, or similar logic whose intent is not obvious from names alone.
- For key interface functions, public dataset/model/pipeline entrypoints, and helper functions whose behavior or parameters are not immediately obvious from the signature alone, add or update docstrings instead of leaving the interface undocumented.
- Do not treat a helper as self-documenting merely because it is private.
  Private helpers that own validation, conversion, serialization restore,
  sequence-field alignment, model/processor boundaries, or external-library
  adaptation should have a contract docstring or adjacent comment. Leave
  simple local helpers undocumented when the signature and body are obvious.
- For private classes or functions that own a key pipeline stage, input
  contract validation, copy/mutation ownership, random or stateful policy, or
  failure semantics, write a complete contract docstring even when the symbol
  is not public. Cover ownership, input/output semantics, whether the original
  object is mutated or copied, and the important exception conditions.
- For key public classes/functions and boundary helpers where safe use is not
  obvious from the signature alone, keep docstrings contract-first and
  user-task-first. Start by explaining what the caller is trying to do and why
  they would use the interface, then document ownership, lifecycle, and other
  caller-visible constraints. Use
  `.agents/templates/interface-docstring-scaffold.md` only when a short
  drafting scaffold is helpful.
- Follow the project's existing Google-style docstring format with `Args:` and `Returns:` when documenting functions.
- In `Args:`, use `name (Type): ...` for required parameters and `name (Type, optional): ... Default is ...` for optional parameters.
- Keep docstrings concise, with consistent indentation and defaults documented only for optional parameters.
- For public Python docstrings that render into API docs, prefer standard
  Sphinx / Napoleon-friendly section shapes. Do not introduce ad hoc section
  headers followed by lists or examples unless the local docs toolchain
  already uses that pattern safely.
- Prefer `Example::`, standard `Examples:` sections, or explicit
  `.. code-block:: python` blocks for usage examples in public docstrings.

## Dependencies

- Avoid new dependencies unless they are clearly necessary.
