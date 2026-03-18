# AGENTS.md

This file defines repository-scoped instructions for Codex and other coding
agents working anywhere under this repository tree.

## Primary Instruction Source

- Treat `.github/instructions/` as the primary human-authored instruction
  source for this repository.
- Before making changes, read the instruction files in
  `.github/instructions/` that are relevant to the task.
- At minimum, always read:
  - `.github/instructions/default.instructions.md`
- Also read these when applicable:
  - `.github/instructions/environment.instructions.md`
    - when the task depends on the active Python environment, optional extras,
      external services, hardware, display access, or other runtime-specific
      conditions
  - `.github/instructions/python.instructions.md`
    - when modifying Python source files, tests, packaging metadata, or
      implementation-related documentation
  - `.github/instructions/workflow.instructions.md`
    - when validating changes or working with repository workflows, tests,
      documentation builds, or developer tooling

## How To Apply Them

- Follow `.github/instructions/` as the source of truth for repository rules.
- Do not duplicate or reinterpret those instructions unless required to resolve
  a direct conflict with higher-priority system, developer, or user guidance.
- If this file and `.github/instructions/` ever diverge, prefer updating this
  file to point at the instruction files rather than copying their contents
  here.

## Repository-Specific Bridging Notes

- Use `Makefile` as the workflow source of truth when relevant targets exist.
- Use `pyproject.toml` as the Python, packaging, and tool configuration source
  of truth.
- Use `tests/pytest.ini` and test-local config as the pytest source of truth.
- Prefer source files over generated copies under `build/`.
- Do not modify files under `build/` unless the task explicitly targets build
  outputs.

## Agent Behavior

- Reply in the same language as the user's request unless they explicitly ask
  for another language.
- Prefer the smallest viable change and avoid unrelated refactoring.
- Before editing, read the relevant implementation, nearby call sites, existing
  tests, and the applicable files under `.github/instructions/`.
- When reporting results, clearly separate completed changes, remaining risks,
  and optional follow-up work.
