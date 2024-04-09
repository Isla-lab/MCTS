from typing import Any

import numpy as np
from copy import deepcopy
from agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from agents.parameters.mcts_parameters import MctsParameters
from utils.mcts_utils import my_deepcopy


class MctsHash(AbstractMcts):
    """
    MonteCarlo Tree Search Vanilla which uses
    """

    def __init__(self, param: MctsParameters):
        super().__init__(param)
        self.root = StateNodeHash(
            data=param.root_data,
            param=param
        )

    def fit(self) -> int | np.ndarray:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """
        initial_env = deepcopy(self.param.env)

        for s in range(self.param.n_sim):
            self.param.env = deepcopy(initial_env)
            self.root.build_tree_state(self.param.max_depth)

        # compute q_values
        self.q_values = np.array([node.q_value for node in self.root.actions.values()])

        # return the action with maximum q_value
        max_q = self.q_values.max()

        # get the children which has the maximum q_value
        max_children = list(filter(lambda c: c.q_value == max_q, list(self.root.actions.values())))
        policy: ActionNodeHash = np.random.choice(max_children)
        return policy.data


class StateNodeHash(AbstractStateNode):

    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.visit_actions = np.zeros(param.n_actions)

    def build_tree_state(self, curr_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """
        # SELECTION
        # to avoid biases if there are unvisited actions we sample randomly from them
        if 0 in self.visit_actions:
            # random action
            action = np.random.choice(np.flatnonzero(self.visit_actions == 0))
            child = ActionNodeHash(data=action, param=self.param)
            self.actions[action] = child
        else:
            action = self.param.action_selection_fn(self)
            child = self.actions.get(action)
        reward = child.build_tree_action(curr_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += reward

        return reward


class ActionNodeHash(AbstractActionNode):

    def build_tree_action(self, curr_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _, _ = self.param.env.step(self.data)

        # if the node is terminal back-propagate instant reward
        if terminal:
            # print('It\'s terminal')
            state = self.children.get(str(observation), None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                state.terminal = True
                self.children[str(observation)] = state
            # ORIGINAL
            self.total += instant_reward
            self.na += 1
            # MODIFIED
            state.ns += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(str(observation), None)
            if state is None:
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                self.children[str(observation)] = state
                # ROLLOUT
                delayed_reward = self.param.gamma * state.rollout(observation, curr_depth + 1)

                # BACK-PROPAGATION
                self.na += 1
                state.ns += 1
                self.total += (instant_reward + delayed_reward)
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                delayed_reward = self.param.gamma * state.build_tree_state(curr_depth + 1)

                # # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                return instant_reward + delayed_reward
