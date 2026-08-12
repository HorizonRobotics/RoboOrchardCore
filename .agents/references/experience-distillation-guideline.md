# Experience Distillation Guideline

Use this reference after
`.agents/instructions/experience-distillation.instructions.md` routes the task
here.

This file describes how to turn completed implementation or review work into
durable local guidance without polluting shared assets with task-local noise.

## Promotion Test

Promote a lesson into shared guidance only when all of these are true.

- The lesson has been validated by a completed implementation, review, or
  repeated task pattern.
- The lesson is likely to recur beyond one file, one module, or one
  temporary branch.
- The lesson can be stated as a stable rule, checklist item, routing rule, or
  scaffold field without preserving the one-off task history.
- The right storage layer is clear enough that future tasks can discover it.

If any of these fail, keep the note out of local shared guidance.

## Candidate Dimensions

Consider each of these dimensions before deciding that there is nothing
durable to capture:

- best practices
- methodology
- project decision records
- terminology
- interface boundaries
- anti-patterns
- test strategy
- migration strategy
- naming conventions
- tool or workflow experience

## Which Layer To Update

- Update `AGENTS.md` when repository scope, precedence, or routing
  entrypoints change.
- Update an instruction when the applicability test for a guidance family
  changes, or when agents need a new read-first path.
- Update a reference when stable rules, checklists, invariants, terminology,
  or anti-patterns change.
- Update a template when repeated planning or review structure emerges and a
  reusable fill-in scaffold would reduce drift.
- Update a skill when reusable workflow-specific execution logic or reporting
  expectations change.

## Local Guidance Versus Memory

Treat tracked repository assets as the durable repository-owned memory.
External agent/session memory is useful, but it may not follow this repository
into another checkout or execution environment.

- Prefer `.agents`, `docs/`, package docs, or code-near TODOs when the lesson
  is a stable repository rule, interface boundary, workflow requirement,
  validation checklist, or decision that future maintainers should inherit
  with the repository.
- Prefer memory when the lesson is reusable context for future agent sessions
  but too local for tracked guidance, such as environment quirks, one-off
  branch state, operator preferences, exact successful commands, or older
  rollout evidence.
- Use both when appropriate: local guidance carries the durable rule, while
  memory can preserve task-specific evidence, exact commands, or historical
  context that would make tracked guidance noisy.
- Before deciding that memory coverage is enough, explicitly ask whether the
  stable part should be represented in tracked local guidance so it survives
  environments without that memory.

## Placement Decision Questions

Before editing guidance, ask:

- Who will consume the lesson?
- Is the lesson a rule, principle, workflow, template field, or historical
  decision?
- Is the lesson repository-wide, skill-family-local, or domain-specific?
- Does an authoritative file already exist?
- Would the likely destination become mixed-scope, repetitive, or harder to
  route if this lesson were added there?

If a lesson fits multiple places, put the durable rule in the narrowest
reusable file and add only short routing pointers where discoverability
requires them.

## Candidate Report Shape

When reporting candidate distillation items before editing, include:

- proposed content
- recommendation, such as apply now, defer, no-distill, or already covered
- recommended destination
- why it is worth keeping
- whether it should be applied now or left as a candidate
- any existing guidance it may overlap with
- whether memory should also be updated, and why tracked local guidance is or
  is not the right durable destination
- user decision, initially pending unless the user already made an explicit
  choice
- agent objection, when the user's requested destination or action would make
  guidance less accurate, less reusable, or unsafe to publish

When the input is a retrospective list with more than a few items, prefer a
coverage matrix before the final recommendation:

| Lesson | Recommendation | Destination | Existing Coverage | Reason | User Decision | Agent Objection |
| --- | --- | --- | --- | --- | --- | --- |

Every original lesson should appear exactly once, or be intentionally merged
with a nearby row. If a lesson is not recommended for guidance or memory,
state whether it is already covered, task-local, unvalidated, or deferred.

Recommendations should come before user decision. If the user chooses a path
that conflicts with the promotion tests or repository publication rules, stop
and state the objection instead of silently applying the update.

## Update Loop

1. Capture the candidate lesson in a scaffold or scratch note.
2. Decide whether the lesson is repository-wide, skill-family-local, or
   domain-specific.
3. Choose the asset layer that future tasks should load first.
4. Update the instruction, reference, template, or skill files that changed.
5. Remove repeated scope text from downstream assets when routing now owns
   that responsibility.

## Temporary Document Cleanup Checkpoint

If temporary design notes or temporary development docs were used during the
task, treat deletion as the last step rather than routine cleanup.

- Get explicit user confirmation before deletion.
- Perform the distillation pass first.
- Update the appropriate instruction, reference, template, or skill files
  when a durable lesson exists.
- Keep one-off notes out of shared guidance when the lesson does not justify
  an asset edit; record memory only for reusable context that remains useful
  without becoming tracked repository guidance.

## Anti-Patterns

- Do not copy task transcripts into shared guidance.
- Do not encode temporary implementation compromises as durable rules.
- Do not encode one-off user collaboration preferences or turn-local workflow
  requests as shared guidance unless this repository explicitly adopts them
  as stable policy.
- Do not update references or templates when the real change is routing.
- Do not add a new guidance family when updating an existing one would keep
  discovery simpler.
