---
description: Load these instructions when validating changes or working with repository workflows, tests, documentation builds, or developer tooling.
# applyTo: 'Makefile' # Expand this if you want automatic loading for tests, pyproject.toml, CONTRIBUTING.md, or other workflow-related files.
---

# Workflow and Validation Instructions

## Project Context
- Use `Makefile` as the workflow source of truth when targets exist.
- Use `pyproject.toml` as the Python and tool configuration source of truth.
- Report inconsistencies between workflow or config files instead of guessing.
- Prefer source files over generated copies under `build/`.

## Source of Truth Priority
- Resolve overlaps in this order:
  1. `Makefile`
  2. `pyproject.toml`
  3. `robo_orchard_core/`
  4. `tests/` and `test/`
  5. `build/` for debugging only
- Do not infer behavior or structure from `build/` when matching source files exist.

## Testing
- Add or update tests when business logic changes.
- Pick validation from the changed files and impact area.
- Start with the smallest sufficient command.
- Build targeted `ruff`, `pyright`, `pytest`, or formatting commands from `pyproject.toml`, `pytest.ini`, and related config.
- Do not default to repository-wide commands when file-scoped or module-scoped validation is enough.
- Expand validation only for shared behavior, public APIs, packaging, cross-module flows, or repository-wide config changes.
- State why the chosen validation scope is sufficient.
- Use this default mapping unless the change clearly needs more or less:
  - Python implementation changes: targeted `ruff`, targeted `pyright` when typing may change, and related `pytest`.
  - Test-only changes: targeted `pytest`, plus lint only for modified files when needed.
  - Documentation or comment-only changes: lightweight validation only.
  - `Makefile`, `pyproject.toml`, or test config changes: broaden validation.
- Tests should cover:
  - the normal path
  - boundary conditions
  - failure paths
- If only part of the validation runs, state what was covered and what remains unverified.
- If validation cannot run, state the reason, risk, and recommended follow-up.
- Do not delete tests unless they are confirmed obsolete and the reason is documented.
- Do not assume optional services or tools are available; call out preconditions such as `ray`.
- If using reduced validation for docs, comments, or formatting-only changes, say so explicitly.

## Git Commit Convention
- Use this format for commit messages: `<type>(<scope>): <Description>.`
- Use one of these types: `feat`, `fix`, `bugfix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `scm`.
- Keep `<scope>` short and directly related to the changed area.
- Keep `<Description>` concise, imperative, specific, and end it with a period.
- Keep the full commit message within 128 characters.
- Example: `feat(api): Implement new authentication module.`

## GitLab Merge Request Description
- This section applies only when drafting a GitLab merge request description; do not assume the same format for GitHub pull requests or other review systems.
- When asked to draft a GitLab merge request description, compare the current branch against `master` unless the user specifies a different target branch.
- The first line must be the proposed squash commit message and must follow the Git commit convention above, including the trailing period.
- The proposed squash commit message must stay within the same 128-character limit.
- The proposed squash commit message must summarize the full branch diff against the target branch, not just the most recent commit.
- Do not reuse the latest commit message as the squash commit message unless the latest commit alone accurately represents the entire branch diff.
- When the branch contains multiple related commits, choose a scope and description that cover the combined user-facing and developer-workflow impact of the branch.
- After the first line, provide the GitLab merge request description as separate prose, not as part of the commit message.
- Write GitLab merge request descriptions in English by default unless the user explicitly requests a different language.
- Summarize the change set from the actual diff, including the main behavior changes, developer workflow updates, and any validation that was run.
- Keep the description factual and scoped to the branch diff; do not mention changes that are not present in the comparison.

## Review Checklist
- Confirm the change matches the request and stays in scope.
- Confirm unrelated behavior did not change.
- Confirm relevant validation ran when practical.
- Confirm validation scope matches changed files and impact.
- Confirm `build/` was untouched unless the task required it.
- Confirm assumptions, residual risks, and follow-up items are explicit.
