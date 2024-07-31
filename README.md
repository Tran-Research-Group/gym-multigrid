# gym-multigrid

This repo is intended to be a lightweight, multi-agent, gridworld environment. It was originally based on the [minigrid-inspired multigrid environment](https://github.com/ArnaudFickinger/gym-multigrid), but has since been heavily modified and developed beyond the scope of the original environment. Please cite this respository as well as the original repository if you use this environment in any of your projects:

```
@misc{multigrid,
  title = {Tran Research Group Gridworld Environment},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Tran-Research-Group/trg-multigrid}},
}
```

## Getting started
This repo uses [poetry](https://python-poetry.org/docs/) library dependency management. To install the dependencies for this project run:
```
poetry install
```

## Included environments
### Capture-the-Flag (CtF)
### Collect
### Maze
### Wildfire

## Extending multigrid
Please see this [guide](https://docs.google.com/document/d/13bCjSzRvLkdGWx7er67VQwF87pJmRIkDR41fm6iMToI/edit?usp=sharing) for creating a custom multigrid environment and CONTRIBUTING.md for code guidelines if you are interested in adding your environment to this repo.
