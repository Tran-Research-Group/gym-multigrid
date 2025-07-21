from typing import Generic, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict

Position: TypeAlias = tuple[int, int]
Size: TypeAlias = tuple[int, int]

EnvKwargs = TypeVar("EnvKwargs")


class EnvMakeConfig(BaseModel, Generic[EnvKwargs]):
    id: str = "multigrid-rooms-v0"
    max_episode_steps: int | None
    disable_env_checker: bool | None = None
    env_kwargs: EnvKwargs

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)
