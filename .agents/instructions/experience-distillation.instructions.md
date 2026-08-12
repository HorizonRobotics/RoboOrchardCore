---
description: Use this guidance when distilling stable implementation, review, or design-process lessons into local guidance assets or memory in this repository.
---

# Experience Distillation Instruction

Use this instruction for local guidance and memory distillation work in this
repository.

## Distillation Trigger

Run a distillation pass when the user asks for "沉淀", "复盘",
"总结到 .agents", "有什么可以沉淀", or equivalent wording after
implementation, design, review, debugging, or workflow iteration.

## Procedure

- For non-trivial distillation, use
  `.agents/references/experience-distillation-guideline.md` as the decision
  checklist and `.agents/templates/experience-distillation-template.md` as the
  candidate scaffold.
- For long-running, resumed, or heavily iterative tasks, the structured
  capture pass is required. Complete the template's Conversation Coverage,
  Asset-Type, and Skill / Workflow passes before filtering candidates by
  destination. The capture may remain in the active plan or report; it does
  not need to become a standalone tracked artifact.
- If the user asks what can be distilled or whether guidance should change,
  report candidate lessons and destinations first unless they explicitly ask
  for immediate edits.
- For long-running, resumed, or heavily iterative tasks, evaluate the whole
  current conversation when identifying distillation candidates. Include
  design decisions, review feedback, debugging outcomes, implementation
  corrections, validation choices, and git workflow lessons instead of only
  inspecting the latest design note or code diff.
- When distillation is triggered as a workflow gate before commit, human
  review handoff, or scratch-design retirement, still evaluate the whole
  current conversation rather than only the final diff, latest document, or
  last review round.
- Before editing durable guidance or memory, report candidate lessons with an
  agent recommendation such as apply now, defer, no-distill, or already
  covered. Include the proposed destination and reason, then wait for the
  user's decision unless the user has already explicitly approved that exact
  update.
- If the requested update would encode an unvalidated opinion, bury a
  task-local detail in shared guidance, conflict with existing guidance, leak
  non-public information into public-facing text, or preserve a transcript
  instead of a reusable rule, pause and explain the objection. Proceed only if
  the user provides new facts or reasoning that resolves the objection.
- When reporting candidates from a broad session, classify them by intended
  asset type, such as instruction/workflow rule, reference/guideline,
  template/scaffold, memory note, implementation-validated follow-up, or
  task-local detail that should stay out of shared guidance.
- When a retrospective lesson list already exists, keep later distillation
  recommendations traceable to that list. For non-trivial lists, use a
  coverage matrix so every lesson is mapped to guidance, memory, a deferred
  decision, or an explicit no-distill reason.
- When the user asks to distill or preserve lessons, explicitly evaluate both
  tracked local guidance and repo/session memory. Do not treat memory as a
  substitute for repository-owned guidance: if the lesson should follow this
  repository into another environment, promote it to tracked `.agents`,
  `docs/`, package docs, or code-near TODOs first. Use memory only for
  reusable context that is helpful to future agents but too local,
  environment-specific, or task-specific for tracked guidance.
- Update assets by layer:
  - `AGENTS.md` for scope, precedence, and routing only
  - `.agents/instructions/*.md` for applicability and read-first behavior
  - `.agents/references/*.md` for stable rules, checklists, and decision summaries
  - `.agents/templates/*.md` for reusable agent-facing output skeletons,
    intake forms, report structures, and scaffold documents
  - `.agents/skills/*` for reusable workflow-specific execution logic or reporting expectations
- During distillation, re-evaluate asset shape as well as content. If the
  lesson deserves a dedicated asset or a cleaner split, follow the extraction
  rules in `.agents/instructions/guidance-authoring.instructions.md` instead
  of automatically appending it to the file already in hand.
- Prefer updating an existing local guidance family when the lesson is
  clearly tied to an established domain or workflow in this repository.
- Promote a lesson into local shared guidance only when it is validated,
  likely to recur, and discoverable from the local `AGENTS.md` routing.
- If the task used temporary design notes or temporary development docs,
  finish the distillation pass before those documents are deleted.
- Do not promote one-off bug fixes, temporary drafts, or unvalidated opinions
  into shared guidance.
- Do not promote turn-local collaboration preferences or one-off workflow
  requests into shared guidance unless this repository explicitly adopts them
  as stable policy.
