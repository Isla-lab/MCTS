from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import gym


@dataclass(slots=True)
class MctsParameters:
    # data of the root node
    root_data: Any
    # simulation environment
    env: gym.Env | None
    # number of simulations from root node
    n_sim: int
    # exploration-exploitation factor
    C: Union[float, int]
    # the function to select actions
    action_selection_fn: Callable
    # discount factor
    gamma: float
    rollout_selection_fn: Callable
    # # the name of the variable containing state information inside the environment
    # state_variable: str
    #  max number of steps during rollout
    max_depth: int
    # number of available actions
    n_actions: Union[None, int]
    depths: list
