---
description: Load these instructions when planning complex repository work, validating changes, or working with repository workflows, tests, documentation builds, or developer tooling.
---

# Workflow and Validation Instructions

## Sources of Truth

- Use `Makefile` when a relevant target exists.
- Use `pyproject.toml` and pytest config for tool behavior.
- Prefer source files over `build/`; use `build/` only for debugging generated output.
- If workflow files disagree, report the mismatch instead of guessing.

## Design And Delivery Loop

- For complex, cross-cutting, or high-uncertainty tasks, follow this minimal loop: design, develop, confirm, distill, then clean up.
- When the `feature-dev` skill applies, treat it as the detailed implementation of the design, develop, and confirm portions of this loop. The repository-level distill and clean-up requirements still apply after that skill's development flow completes.
- Before implementation, write a temporary design note in a disposable repository-local scratch path such as `.agents/temp/designs/`; keep it uncommitted by default.
- Capture the problem, constraints, chosen approach, validation plan, and explicit non-goals in that temporary design note.
- Skip the temporary design note for small, local, or mechanical changes when the implementation path is already obvious.
- After implementation, run the smallest useful validation and confirm the result against the user's request before treating the task as complete.
- After confirmation, delete the temporary design note.
- If part of the temporary design is durable project knowledge, promote only the stable subset into this repository's canonical design docs, `docs/`, package docs, or another established design-doc location instead of preserving the scratch note.
- If the work reveals durable agent-facing lessons, distill them into local guidance or other intentional local shared agent assets instead of copying the whole temporary design note into instructions.

## Validation

- Choose the smallest validation that matches the changed files and impact.
- Add or update tests when behavior changes.
- Broaden validation for shared behavior, public APIs, packaging, or config changes.
- If validation is partial or blocked, say what ran, what did not, and the remaining risk.
