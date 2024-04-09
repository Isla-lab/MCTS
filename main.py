import argparse
from tqdm import tqdm
import time
import gymnasium as gym
import numpy as np
from agents.mcts_hash import MctsHash
from agents.parameters.mcts_parameters import MctsParameters
from utils.action_selection_functions import ucb1, discrete_default_policy


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nsim', default=500, type=int, help='The number of simulation the algorithm runs')
    parser.add_argument('--c', default=5, type=float, help='exploration-exploitation factor')
    parser.add_argument('--as', default="ucb", type=str, help='the function to select actions during simulations')
    parser.add_argument('--ae', default="random", type=str, help='the function to select actions to add to the tree')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--rollout', default='random', type=str, help='the function to select actions during rollout')
    parser.add_argument('--max_depth', default=500, type=int, help='max number of steps during rollout')
    parser.add_argument('--n_episodes', default=100, type=int,
                        help='the number of episodes for each experiment')
    return parser


def get_function(function_name):
    dict_args = args.__dict__
    functions = {
        "ucb": ucb1,
        "random": discrete_default_policy,
    }
    return functions[function_name]


def create_agent(env, root_data):
    dict_args = args.__dict__
    param = MctsParameters(
        root_data=root_data,
        env=env,
        n_sim=dict_args["nsim"],
        C=dict_args["c"],
        action_selection_fn=get_function(dict_args["as"]),
        rollout_selection_fn=get_function(dict_args["rollout"]),
        gamma=dict_args["gamma"],
        max_depth=dict_args["max_depth"],
        n_actions=env.action_space.n,
        depths=[]
    )
    return MctsHash(param)


def main():
    dict_args = args.__dict__

    times = []
    rewards = []
    for _ in tqdm(range(dict_args['n_episodes'])):
        real_env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
        observation = real_env.reset()
        done = False
        start_time = time.time()
        while not done:
            agent = create_agent(real_env, observation)
            action = agent.fit()
            observation, reward, done, _, _ = real_env.step(action)
            rewards.append(reward)
            agent.param.root_data = observation
        times.append(time.time() - start_time)
    print(f"Time: {np.mean(times)}")
    print(f"Terminal runs: {int(np.sum(rewards))} / {dict_args['n_episodes']}")


if __name__ == '__main__':
    global args
    args = argument_parser().parse_args()
    main()
