# Contribution Guide

## Install by editable mode

```bash
make install-editable
```

## Install development requirements

```bash
make dev-env
```

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

make sure that ray is installed and running before executing the following command.
If ray is not running, you can start it by running `ray start --head` in your terminal.

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing robotics frameworks, such as [OpenAI Gym](https://gym.openai.com/), [Robosuite](https://robosuite.ai/docs/index.html), [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
