# Experience Distillation Template

Use this template when deciding whether completed work should update shared
or domain-local instructions, references, or templates.

## Task Snapshot

- Task or feature: `<name>`
- Domain: `<dataset / model / deployment / workflow / other>`
- Trigger: `<implementation / review / regression / design iteration>`
- Validation signal: `<tests / review findings / repeated occurrence>`

## Conversation Coverage Pass

Review every evidence area before selecting destinations. Record a candidate
or an explicit `none` so the final diff or latest technical topic cannot
silently narrow a long-running task.

| Evidence Area | Candidate Or None | Source Or Validation |
| --- | --- | --- |
| Design decisions, ownership boundaries, and rejected alternatives |  |  |
| Implementation corrections and architecture convergence |  |  |
| Review findings, accepted fixes, and rejected or deferred feedback |  |  |
| Tests, profiles, remote jobs, environment routing, and failure recovery |  |  |
| Human review gates, agent delegation, monitoring, and handoff workflow |  |  |
| Git, commit, push, branch, and dirty-worktree boundaries |  |  |
| Documentation synchronization, scratch notes, and retirement |  |  |
| Task-local details that should not become durable guidance |  |  |

## Asset-Type Pass

Classify by asset type before grouping by domain topic. Include
instruction/workflow-rule candidates explicitly, even if the final decision is
not to edit an instruction.

- Instruction or workflow-rule candidates:
  `<files or none; reason>`
- Reference or guideline candidates:
  `<files or none; reason>`
- Template or scaffold candidates:
  `<files or none; reason>`
- Skill or skill-local candidates:
  `<files or none; reason>`
- Memory candidates:
  `<memory note or none; why tracked project guidance is or is not needed>`
- Task-local details to keep out of shared guidance:
  `<items and reason>`

## Skill / Workflow Impact

- Affected skill workflow:
  `<SKILL.md / skill-local reference / none>`
- Trigger or routing impact:
  `<yes/no and details>`
- Execution-step impact:
  `<yes/no; reviewer roles, pass ordering, validation steps, or handoff>`
- Report or template impact:
  `<yes/no and affected report/template>`
- Reference-only rationale:
  `<why SKILL.md does not need to change, if no skill edit is proposed>`

## Candidate Lessons

| Candidate | Dimension | Proposed Content | Recommended Destination | Memory? | Why Keep It | Apply Now? | Overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lesson 1 | `<best practice / methodology / decision record / terminology / interface boundary / anti-pattern / test strategy / migration strategy / naming convention / workflow experience>` |  |  | `<yes/no and whether tracked guidance should also change>` |  |  |  |

## Promotion Test

- Is the lesson validated beyond one speculative draft?
- Is it likely to recur?
- Can it be written without one-off task history?
- Which future tasks should load it?
- Who will consume it?
- Is it repository-wide, skill-family-local, or domain-specific?
- Does an authoritative guidance file already exist?

## Asset Decision

- Update `AGENTS.md`:
  `<yes/no and why>`
- Update instruction:
  `<file and why>`
- Update reference:
  `<file and why>`
- Update template:
  `<file and why>`
- Update memory:
  `<yes/no; why memory is useful and why tracked local guidance is or is not needed>`
- No promotion:
  `<which lessons should stay out of shared assets>`

## Proposed Wording

- Routing or applicability wording:
  `<text>`
- Stable rule or checklist wording:
  `<text>`
- Scaffold field changes:
  `<text>`

## Follow-Through

- Files to edit:
  `<paths>`
- Repeated wording to remove after routing is updated:
  `<paths or phrases>`
- Remaining open questions:
  `<questions>`
