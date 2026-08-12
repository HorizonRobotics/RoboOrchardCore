---
description: Load these instructions when planning complex repository work, validating changes, or working with repository workflows, tests, documentation builds, or developer tooling.
---

# Workflow and Validation Instructions

## Sources of Truth

- Use `Makefile` when a relevant target exists.
- Use `pyproject.toml` and pytest config for tool behavior.
- Prefer source files over `build/`; use `build/` only for debugging generated output.
- If workflow files disagree, report the mismatch instead of guessing.

## Documentation Builds

- When Python changes affect rendered documentation, API docs, examples, or
  doctest-like snippets, prefer `make doc` as the broad documentation
  validation path when the target exists.
- Keep documentation build instructions aligned with the repository
  `Makefile`, `docs/` configuration, and `pyproject.toml` instead of adding
  ad hoc commands that bypass the canonical workflow.

## Design And Delivery Loop

- For complex, cross-cutting, or high-uncertainty tasks, follow this minimal loop: design, develop, confirm, distill, then clean up.
- When the `feature-dev` skill applies, treat it as the detailed implementation of the design, develop, and confirm portions of this loop. The repository-level distill and clean-up requirements still apply after that skill's development flow completes.
- Before implementation, write a temporary design note in a disposable
  repository-local scratch path such as `.agents/scratch/designs/`; keep it
  uncommitted by default.
- For complex refactors, cross-layer features, public API changes, resource
  lifecycle changes, schedulers, or compatibility migrations, consult
  `.agents/references/design-doc-guideline.md`. Use
  `.agents/templates/design-doc-scaffold.md` only as an optional prompt list;
  do not force the note into that order.
- Capture the problem, constraints, chosen approach, validation plan, and explicit non-goals in that temporary design note.
- Keep design notes focused on architecture, contracts, failure modes,
  compatibility, and validation. Keep mechanical task checklists or execution
  plans separate unless they clarify the design decision.
- Skip the temporary design note for small, local, or mechanical changes when the implementation path is already obvious.
- Do not bake a commit requirement into a design or implementation plan unless
  the user explicitly asked for a commit.
- For user-reviewed phased development, finish implementation, validation,
  and cleanup for the phase, then stop at a review gate. Run delegated review
  at key checkpoints, such as phase completion with public API, persisted
  format, or boundary changes, or when the user explicitly asks for review.
  Do not commit that phase until the user explicitly says it can be committed.
- For manual review between phases, code, design notes, plans, and review
  findings may move at different speeds during active iteration, but before
  handing work back they must all describe the same current implementation,
  status, validation evidence, and remaining risk.
- At each human-review handoff, identify the current planned review gate and
  the number of planned gates remaining. Report issue-triggered extra review
  rounds as conditional instead of presenting them as fixed planned gates.
- Before committing, handing work back for human review, or retiring a scratch
  design after non-trivial work, run a distillation triage over the whole
  current conversation. Report candidate lessons with recommendations first,
  wait for the user's decision before editing durable guidance or memory, and
  record `none` when there is nothing worth preserving.
- Treat a design as ready for implementation only after the ownership
  boundaries, user-facing contract, compatibility posture, failure behavior,
  and validation scope are clear enough to guide review.
- After implementation, run the smallest useful validation and confirm the result against the user's request before treating the task as complete.
- If part of the temporary design is durable project knowledge, promote only the stable subset into this repository's canonical design docs, `docs/`, package docs, or another established design-doc location instead of preserving the scratch note.
- If the work reveals durable agent-facing lessons, distill them into local guidance or other intentional local shared agent assets instead of copying the whole temporary design note into instructions.
- After confirmation, delete the temporary design note only after any required
  durable promotion or explicit no-durable-content decision is complete.

## Parallel Work Planning

- Before dispatching parallel implementation work, define stable interfaces,
  disjoint write scopes, dependency order, expected validation, and how
  conflicts or review findings will be reported back.
- Keep cross-cutting public API, packaging, or migration decisions local to the
  coordinating agent until the contract is clear enough for independent tasks.

## Validation

- Choose the smallest validation that matches the changed files and impact.
- Add or update tests when behavior changes.
- Broaden validation for shared behavior, public APIs, packaging, or config changes.
- When deleting a package, command, service, example, or other published
  runtime surface, validate in the same turn that CI or install flows,
  surviving runtime entrypoints, examples, tests or resources, and packaging
  metadata no longer reference the removed surface.
- For broad surface removals, prefer deleting surviving wrappers or entrypoints
  together with the obsolete implementation instead of leaving a partially
  removed surface that still publishes stale imports.
- For broad pytest validation, prefer the repository's canonical entrypoints
  or an explicit xdist worker count that fits the target. Avoid `-n auto`
  unless the repository explicitly documents it for that command.
- Include documentation validation when code changes affect rendered docs, API
  examples, or documented public interfaces.
- If validation is partial or blocked, say what ran, what did not, and the remaining risk.

## Cleanup Gate

- Before finishing substantial work, inspect the diff stat and scan the
  touched files for stale TODOs, legacy compatibility notes, deprecated names,
  and temporary scaffolding introduced during the task.
- Re-check new public exports, `__all__` entries, public classes, protocols,
  and helper modules so the final API surface matches the intended design;
  keep them only when they have a clear caller, compatibility purpose, or
  semantic boundary that cannot be handled by an existing interface.
- When delegating review for non-trivial Python changes that add or modify
  package roots, explicitly ask reviewers to check `__init__.py` and
  `__all__` root-surface minimality.
- Prefer deleting, merging, or downgrading compatibility-only code before
  adding another abstraction to explain it.
- For non-trivial implementation, run an explicit convergence pass after
  correctness findings are closed and before human review. Inventory added or
  materially changed public surfaces, helpers or wrappers, compatibility
  branches, validation or fallback paths, and test scaffolding. For each group,
  delete or merge it, or record its exact caller, compatibility commitment, or
  distinct owner/failure boundary; compare direct/no-new-helper and
  single-flow alternatives.
- If convergence changes executable code, rerun relevant validation, repeat
  the inventory, and complete a final correctness regression scan. Record the
  resulting convergence evidence in the active plan or human-review handoff;
  a one-pass helper scan is not a passed convergence gate.

## External CI Diagnosis

- When external CI logs contain both a primary build/test failure and later
  post-action, sandbox, or log-processing errors, identify the first real
  failure before acting on the tail error.
- For flaky-test diagnosis, inspect the repository's real parallel test
  entrypoints, xdist mode, worker counts, and `Makefile` targets before
  treating an isolated local rerun as representative.
