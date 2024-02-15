import cvxpy as cp

from sperl.data.occupancy import OccupancyMeasure


class ExactOptProblem:
    def __init__(self, setting, meta_env):
        """
        Initializes the optimization problem, including the values for lambda,
        the environment and gamma
        """
        self.lamda = setting.lamda
        self.env = meta_env
        self.gamma = setting.gamma

    def solve(self, prob_trans_and_reward, policy_index):
        """
        Args:
            prob_trans_and_reward -- this is a tuple (prob_trans, reward_func) such
            that prob_trans is a numpy array of shape S x A x S which stores a
            probability transition function. reward_func is a numpy array of
            shape S x A, which stores the reward function.
        """
        prob_transition, reward_func = prob_trans_and_reward

        d = cp.Variable(
            (
                self.env.num_states(policy_index),
                self.env.num_actions(policy_index),
            ),
            nonneg=True,
        )

        objective = cp.Maximize(
            cp.sum(cp.multiply(d, reward_func))
            - self.lamda / 2 * cp.power(cp.pnorm(d, 2), 2)
        )

        constraints = []
        for s in range(self.env.num_states(policy_index)):
            if self.env.is_terminal(s):
                continue
            constraints.append(
                cp.sum(d[s])
                == self.env.start_distribution(policy_index)[s]
                + self.gamma * cp.sum(cp.multiply(d, prob_transition[:, :, s]))
            )

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5)
        return OccupancyMeasure(d.value)
