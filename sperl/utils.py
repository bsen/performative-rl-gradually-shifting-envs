import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from typing import Any, Callable, Dict, List, cast

import numpy as np

from sperl.metaenvs.abstract_env import Environment
from sperl.metaenvs.abstract_meta_env import MetaEnvironment

PATH_BASE = "/local"


def set_random_seed(random_seed=None):
    """Using random seed for python and numpy."""
    if random_seed is None:
        random_seed = 13
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    return


def policy_from_occupancy(occupancy):
    """
    Calculate the policy from a target occupancy measure.
    Args:
        occupancy -- an array from the form num_states x num_actions
    Returns:
        A policy, represented by a numpy array with shape
        num_states x num_actions.
        Each entry policy[s, a] represents the probability of choosing
        action a in state s.
    """
    num_states, num_actions = occupancy.shape
    state_occupancy = np.sum(occupancy, axis=1)

    # the following lines ensure that for the states where the occupancy
    # measure is zero get a value of 1/A in the final policy
    zero_occupancy = np.zeros(num_states)
    zero_occupancy[state_occupancy == 0] = 1
    state_occupancy[state_occupancy == 0] = num_actions
    policy = np.outer(zero_occupancy, np.ones(num_actions))
    policy += occupancy
    # we can then divide this by
    denominator = np.outer(state_occupancy, np.ones(num_actions))
    return policy / denominator


# code from
# https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def exact_value_iteration(env: Environment, gamma, tol=1e-5):
    """
    Given an environment and a gamma value, returns the optimal state-value
    function of this environment.
    """
    # initialize all state values to zero
    values = np.zeros(env.num_states, dtype="float64")
    while True:
        values_old = copy.deepcopy(values)
        delta = 0
        for state in range(env.num_states):
            if env.is_terminal(state):
                values[state] = 0
            else:
                actions = env.actions(state)
                bell = np.zeros_like(actions, dtype="float64")
                for action_id in range(len(actions)):
                    prob_reward_state = env.reward_and_next_state(
                        state, actions[action_id]
                    )
                    for prob, reward, next_state in prob_reward_state:
                        bell[action_id] += prob * (
                            reward + gamma * values_old[next_state]
                        )
                values[state] = np.max(bell)
            delta = max(delta, abs(values[state] - values_old[state]))

        if delta < tol:
            break

    return values


def gen_one_traj(
    random_generator,
    num_states,
    num_actions,
    start_distribution,
    prob_transition,
    reward_func,
    max_sample_steps,
    policy,
    meta_env,
):
    trajectory = []
    # sample one trajectory
    cur_state = random_generator.choice(num_states, p=start_distribution)
    for _ in range(max_sample_steps):
        action = random_generator.choice(num_actions, p=policy[cur_state])
        next_state = random_generator.choice(
            num_states, p=prob_transition[cur_state, action]
        )
        reward = reward_func[cur_state, action]
        trajectory.append((cur_state, action, reward))
        cur_state = next_state
        if meta_env.is_terminal(next_state):
            trajectory.append((next_state, None, None))
            break
    return trajectory


def generate_trajectories_reward_next_state(
    prob_transition,
    reward_func,
    policy,
    num_trajectories,
    max_sample_steps,
    meta_env: MetaEnvironment,
    seed_sequence: np.random.SeedSequence,
) -> List:
    """
    generates a list of trajectories traj of the form
    num_trajectories x len_trajectory x 3, where traj[a,b,c]
    corresponds to trajectory a of the trajectories of this
    round, the b-th entry in this trajectory and if c==0 the state, c==1 the action
    and c==2 the reward of this entry.
    The function assumes that the reward depends on the next state only.

    Args:
        prob_transition -- a probability transition function represented by a numpy
            array of shape S x A x S.
        reward_func -- a reward function represented by a numpy array of shape S.
        policy -- a policy represented by a numpy array of shape S x A, where
            poliy[s, a] represents the probability of choosing action a in state s.
        num_trajectories -- the number of trajectories which is returned
        max_sample_steps -- the maximal trajectory length after which the trajectory
            is cut off.
    """
    seeds = seed_sequence.spawn(num_trajectories)
    start_distribution = meta_env.start_distribution()
    num_states = meta_env.num_states(0)
    num_actions = len(policy[0])
    trajectory_lst = []

    trajectory_lst = [
        gen_one_traj(
            np.random.default_rng(seeds[i]),
            num_states,
            num_actions,
            start_distribution,
            prob_transition,
            reward_func,
            max_sample_steps,
            policy,
            meta_env,
        )
        for i in range(num_trajectories)
    ]

    trajectory_lst = cast(List, trajectory_lst)
    return trajectory_lst


def num_samples(trajectories):
    """
    Args:
        trajectories -- a list of size
            num_trajectories x len_trajectory x 3, where
            trajectories[a,b,c] trajectory a of the
            trajectories of this round, the b-th entry in this trajectory
            and if c==0 the state, c==1 the action and c==2 the reward
            of this entry.
    Returns:
        The number of samples in the list of trajectories.
    """
    num_samples = 0
    for trajectory in trajectories:
        num_samples += len(trajectory) - 1
    return num_samples


def store_data(data, filename):
    file = open(f"{filename}.pickle", "wb")
    pickle.dump(data, file)
    file.close()


def load_data(filename):
    file = open(f"{filename}.pickle", "rb")
    content = pickle.load(file)
    file.close()
    return content


def uniform_policy(num_states, num_actions):
    policy = np.ones((num_states, num_actions))
    policy = np.divide(policy, num_actions)
    return policy


def get_runtime(function: Callable, *args, **kwargs):
    start = time.time()
    function(*args, **kwargs)
    end = time.time()
    return end - start


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
