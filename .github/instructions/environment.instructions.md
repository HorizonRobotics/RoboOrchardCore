---
description: Load these instructions when tasks depend on the active Python environment, optional extras, external services, hardware, or other runtime-specific conditions.
# applyTo: 'pyproject.toml' # Expand this if you want automatic loading for environment-sensitive files.
---

# Environment and Runtime Instructions

## Environment Selection
- Use the active user-selected environment when one is already configured.
- Do not switch Python environments unless the task requires it.
- Do not assume all contributors use the same environment manager.
- Do not assume the active environment includes all optional extras or developer tools.

## Environment Validation
- Check environment-sensitive requirements before running validation that depends on them.
- If a required package, executable, or service is missing, report the gap explicitly.
- Prefer the smallest meaningful validation that fits the active environment.
- If full validation is not possible in the current environment, state what was run, what was blocked, and the remaining risk.

## Runtime Preconditions
- Do not assume network access, hardware devices, display servers, remote services, or background processes are available.
- Call out required runtime preconditions before using commands that depend on them.
- Treat optional services such as `ray` as unavailable until confirmed otherwise.

## Reporting
- Distinguish clearly between not tested, not testable in the current environment, and not required for this change.
- When environment differences affect confidence, state the impact and the recommended follow-up validation.
