---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
# applyTo: '**/*.py' # Expand this if you want automatic loading for pyproject.toml, setup.py, or other Python-related files.
---

# Python Change Instructions

## Python
- Keep changes compatible with the Python version currently used by the project.
- Prefer adding or preserving type annotations for new or modified functions.
- Reuse existing utility functions and established patterns whenever possible.
- Keep new logic small and focused, with a single responsibility per function.
- Avoid magic constants; reuse existing constants or enums when available.
- Do not silently swallow exceptions; when catching exceptions, document the reason and preserve enough context.
- Keep log messages concise, actionable, and free of sensitive information.
- Avoid implicit shared state.
- Preserve backward compatibility for public APIs unless the task explicitly allows breaking changes.

## Code Style
- Follow the repository's existing style instead of introducing a personal style.
- If formatting or static analysis tools are configured, keep the modified code passing them.
- Do not introduce extra abstraction layers solely to make the code look cleaner.
- Write comments only when they add necessary context; do not explain obvious code.
- Keep imports, naming, and file organization consistent with nearby code.

## Dependencies
- Do not introduce new dependencies unless they are necessary for the task.
- When adding a dependency, prefer mature, widely used libraries that fit the current stack.
- Document why the new dependency is needed and what lightweight alternatives were avoided, if relevant.
