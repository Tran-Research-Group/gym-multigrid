## Code conventions
This repository uses the following conventions:
- Docstring style: Numpy
    - The concise description of the function/class should come frist
    - In `Parameters` section, the arguments should be explained.
    - In `Returns`, the outputs should be explained.
    - In `Attributes`, the class attributes should be explained.
    - In `Examples`, the example usage should be explained.
- **Annotate types whenever possible** for building the robust and solid code base.
- Code style: PEP8
- Documentation generation: Sphinx
We highly recommend using [Black](https://black.readthedocs.io/en/stable/) formatting and [pylint](https://pypi.org/project/pylint/).

## GitHub Flow
The basic idea is to create feature branches (each associated with a related issue or contribution) for code modifications. Once the code changes are completed, you merge that branch back with the dev branch. The dev branch should always be functional. The dev branch will be merged with main when the codebase is ready for a release. See [this link](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for more details.

### Branching
1. Create a new branch from `dev` branch.
Name the branch with the following format: `feat/<issue number>_<2-3 words describing the feature>` for features, `fix/<issue number>_<2-3 words describing the fix>` for bug fixes.
2. Make your changes.
3. Create a pull request to `dev` branch.
4. After the pull request is approved, merge it to `dev` branch.
5. After a few features are merged to `dev` branch, create a pull request to `main` branch.

### Setting up the repository
Always remember to pull so your local repo is up to date:
```
git pull origin main
git status
```
Then create a branch from dev and push:
```
git checkout -b new_feature
git push -u origin new_feature
```

### Adding changes
It is good practice to commit your code changes often.
```
git add changed_file.py
git commit -m "added some new code"
```
To push local commits run
```
git push origin new_feature
```

### Merging
Create a pull request for your feature branch to the dev branch and include a summary of what was done and how it was verified. You can request a review from specific people in the side panel.

## Gymnasium
The multigrid environments in this repo inherit [Gymnasium](https://gymnasium.farama.org/). Please see their documentation to understand how best to implement functionality for this repo while adhering to Gymnasium guidelines as well. This [tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#) is a good starting point.

## Testing
We use [pytest](https://docs.pytest.org/en/stable/) for testing.
Each environment should have its own test file in the `tests` directory, and the test file should be named `test_<environment_name>.py`.
There should be the following tests in each test file:
- `test_init`: Test the environment initialization.
- `test_reset`: Test the environment reset. Should return the initial observation and the initial info.
- `test_step`: Test the environment step. Should return the next observation, reward, terminated, truncated, and info.
- `test_render`: Test the environment render. Should return the rendered image.