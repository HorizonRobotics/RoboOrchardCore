---
description: Load these instructions when creating, updating, or validating tests in this repository.
---

# Test Instructions

## Test Design

- Follow the style of nearby tests before introducing a new pattern.
- Prefer the smallest test that proves the target behavior.
- Use real fixtures, datasets, models, and file paths when the test is intended to validate actual integration behavior.
- Do not replace required real test inputs with fallback skip logic when the test is expected to prove correctness in the configured environment.
- When a CLI, plugin, or package import is expected to work after
  installation, test its installed entry surface rather than inferring that
  behavior solely from source-tree paths or packaging metadata.
- Run distribution build and install checks from temporary source and install
  locations rather than the live checkout. Use offline or `--no-deps` modes
  when appropriate, but do not silently skip the artifact proof.
- When a test depends on an external binary, codec, or runtime capability that
  is not a documented repository-baseline requirement, probe that capability
  in the test and skip when unavailable instead of making the default test
  phase fail for environment reasons.
- Use mocks or monkeypatch only when the test target is isolated assembly logic and real dependencies are not part of the behavior under test.
- For import-lightweight tests that manipulate `sys.modules` or optional
  dependency state, prefer a subprocess when Pydantic models, registries, or
  other identity-sensitive globals make an in-process reset unreliable.
- For process-backed or external-binary integrations, prefer one real
  integration path that proves the happy path against the actual dependency
  and separate narrow fake or monkeypatched tests for failure paths that are
  difficult to trigger reliably with the real backend.
- When a wrapper or adapter exposes both raw data and derived data for the same
  contract, test both the assembly logic and at least one end-to-end
  consistency path between the raw and derived representations.
- When extracting shared behavior from existing callers, test the shared unit
  directly and at least one caller boundary path that proves metadata
  reconstruction, pass-through branches, or other caller-owned context still
  survives the refactor.
- For frame or transform adapters, prefer one narrow unit test that checks the
  naming and graph assembly rules and one integration-style test that checks
  numerical consistency against the real runtime source.
- For Pydantic config, reference, or other caller-facing validation surfaces,
  do not rely only on direct Python construction. Prefer covering direct
  construction, `model_validate(...)`, and a JSON round-trip when aliases,
  coercion, or serialized loading behavior matter.
- If branch selection depends on validation or coercion, include at least one
  validation-path test that proves the coerced semantics match the runtime
  branch taken.
- When one config field or kwargs surface fans out to multiple downstream APIs
  with different accepted arguments, add at least one focused negative test
  for an invalid cross-branch combination. Prefer failing at the
  repository-owned boundary with a readable error over relying on a deep
  dependency stack trace as the first signal.

## Fixtures

- Keep reusable fixtures in the nearest `conftest.py` that matches their sharing scope.
- Move shared model paths, tokenizer paths, processor paths, and other reusable test resources out of individual test files when multiple tests in the same directory can reuse them.
- Keep test-specific fixtures in the test module when they are only used by one file.

## Test Structure

- Match the local project convention for test organization; prefer class-based tests when nearby files use `Test...` classes.
- Keep assertions focused on the behavior under test instead of asserting incidental implementation details.
- For backend migrations, refactors, or compatibility-path changes, test the
  intended contract rather than the old implementation path. Assert old
  subprocess calls, adapters, or wrappers only when that path is itself the
  supported contract.
- For import-time registries, optional backend discovery, plugin loading, or
  dependency-gated routing, include at least one import-time test. Prefer a
  subprocess when identity-sensitive globals or already-imported modules make
  in-process reset unreliable.
- When a test is meant to help inspect real returned data, print or otherwise expose the key returned values in the test run so failures and manual verification are easier to interpret.

## Validation

- Before running `python`, `pytest`, or `ruff`, load `.agents/instructions/environment.instructions.md`.
- Run the narrowest relevant `pytest` target for the changed test or module first.
- For small or focused local validation, prefer serial pytest runs. When the
  pytest scope is broad enough that parallelism will materially reduce
  turnaround time, prefer the repository's canonical test entrypoints or an
  explicit worker count that fits the target instead of `-n auto`.
- Do not run simulator, env-rollout, GPU-memory-sensitive, or hardware-backed
  tests under pytest parallelism by default. Keep those tests serial unless
  the task or repository guidance confirms the resource budget and isolation
  are sufficient.
- When running repository tests, disable `HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, and `https_proxy` unless the task explicitly requires proxy access.
- For tests that spawn `accelerate launch`, `torchrun`, or similar
  distributed subprocesses under xdist, avoid probe-and-release free-port
  helpers. Prefer port `0` or another launcher-owned atomic port allocation
  path.
- If a subprocess helper does not need child-process coverage, clear
  inherited `COV_CORE_*` and `COVERAGE_PROCESS_START` variables before launch
  so pytest-cov noise does not mask the real failure.
- Run `ruff check` on modified test files.
- If the local pytest environment requires temporary flags or environment variables to run successfully, document the exact command used and why it was needed.
