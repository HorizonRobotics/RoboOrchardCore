# Contribution Guide

## AI-assisted contribution

- This project is compatible with AI-assisted development, and contributors are encouraged to use GitHub Copilot, Codex, or other coding agents.
- The repository includes AI agent instructions in `AGENTS.md` and `.github/instructions/`; use them as the source of truth when working with AI tools.
- AI-assisted changes should still stay focused, reuse existing patterns, and be reviewed and validated before commit.

## Install by editable mode

```bash
make install-editable
```

## Install development requirements

```bash
make dev-env
```

## Contribution workflow

- Keep changes focused and validate only what matches the scope of the change.
- Use the `Makefile` targets in this document as the default local workflow.
- For commit messages and GitLab merge requests, follow the source-of-truth guidance in `.github/instructions/git.instructions.md`.
- Contributors may use an AI assistant to help stage changes, create commits, push branches, and open merge requests, as long as the resulting git history and MR content follow the repository instructions.
- In particular, keep the commit title in the required `<type>(<scope>): <Description>.` format, include the structured multiline body, and use the same content for the final squash commit message and MR description.

## Lint

```bash
make check-lint
```

## Auto format

```bash
make auto-format
```

### Type checking

```bash
pyright
```

## Build docs

```bash
make doc
```

## Run test

Make sure that Ray is installed and running before executing the following command.
If Ray is not running, you can start it with `ray start --head`.

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing robotics frameworks, such as [OpenAI Gym](https://gym.openai.com/), [Robosuite](https://robosuite.ai/docs/index.html), [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
