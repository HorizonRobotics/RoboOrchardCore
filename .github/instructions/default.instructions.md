---
description: Load these baseline instructions for any task in this repository.
# applyTo: '**/*' # Uncomment if these instructions should be auto-loaded for all files in this repository.
---

# Project Instructions

## General
- Default to replying in the same language as the user's request, unless the user explicitly asks for a different language.
- State the conclusion first, then provide the key reasons.
- Prefer the smallest viable change and avoid unrelated refactoring.
- Before making changes, read the relevant module, its call sites, and existing tests.
- When multiple implementation options exist, recommend the lowest-risk, smallest-change option first.
- For risky operations, explain the risk before giving recommendations.
- If requirements are ambiguous, clarify only the most critical point and avoid high-risk assumptions.

## Scope
- Modify only files that are directly related to the current task.
- Do not perform unrelated refactoring.
- Do not change existing module boundaries unless the task explicitly requires it.
- Do not adjust the project structure without a clear need.
- Do not rename public functions, classes, modules, or configuration keys unless required.
- Do not change unrelated comments, formatting, or import order unless correctness or tooling requires it.
- Do not modify generated or packaged artifacts under `build/` unless the task explicitly targets build outputs.

## Safety
- For file deletion, bulk writes, external system calls, or database changes, warn about the risk first.
- Do not assume the local environment has external services, hardware devices, or network access.
- For potentially breaking changes, clearly describe the impact area first.
- Prefer reversible changes and explicit rollback guidance when a task may affect developer workflow or runtime behavior.

## Output
- When code is modified, list the affected file paths in the response.
- If commands are needed, prefer Linux commands.
- If suggesting tests or checks, list only the minimum necessary commands.
- If information is insufficient, ask only the most critical question and avoid high-risk guesses.
- Summaries should distinguish clearly between completed changes, remaining risks, and optional follow-up work.

## Documentation
- When changing public behavior, configuration, or developer workflow, update the relevant documentation if it is part of the task scope.
- Keep documentation aligned with the actual implementation; avoid speculative or aspirational wording.
- Prefer concise usage notes, constraints, and known limitations over long narrative explanations.