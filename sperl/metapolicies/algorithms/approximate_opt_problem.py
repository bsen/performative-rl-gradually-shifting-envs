import sys
import traceback

import cvxpy as cp
import numpy as np

from sperl import utils
from sperl.data.occupancy import OccupancyMeasure
from sperl.utils import eprint

# pyright: reportGeneralTypeIssues=false


class ApproximateOptProblem:
    SOLVER_KWARGS = {
        "solver": cp.CVXOPT,
        "abstol": 1e-5,
        "reltol": 1e-4,
        "feastol": 1e-5,
    }
    # SOLVER_KWARGS = {"solver": cp.SCS, "eps": 1e-5}

    def __init__(
        self,
        meta_policy,
    ):
        self._meta_policy = meta_policy
        self._meta_env = meta_policy.meta_env

        run_config = meta_policy.run_config
        self.num_ftrl_steps = run_config.num_ftrl_steps
        self.lamda = run_config.regularizer
        self._meta_env = meta_policy.meta_env
        self.gamma = run_config.gamma
        self.regularize_h = run_config.regularizer
        self.random_generator = meta_policy.random_generator
        self.B = run_config.b

    def solve(self, trajectories_over_t, occupancies_over_t, policy_index, initial_d):
        """
        Args:
            trajectories_over_t -- a list of size
                k x num_trajectories x len_trajectory x 3, where
                trajectories_over_t[a,b,c,d]
                corresponds to round a in one iteration, trajectory b of the
                trajectories of this round, the c-th entry in this trajectory
                and if d==0 the state, d==1 the action and d==2 the reward
                of this entry.
            occupancies_over_t -- a list of size k of occupancy measures
                for each round t in {1, ..., k}. (Typically the input is an
                approximation of said occupancy measures.)

        Steps of finding the policy:

        1. Calculate the "Follow the regularized Leader", where the d-player
            is responding with a best response and h-player
            uses a regularized objective
        2. Compute the average of all ds computed in this way
        """
        assert len(trajectories_over_t) == len(occupancies_over_t)
        num_states = self._meta_env.num_states(policy_index)
        num_actions = self._meta_env.num_actions(policy_index)

        # d_val = initial_d

        # the current values of h
        h_val = np.zeros(num_states)
        # the values of past versions of h
        h_vals = np.empty((self.num_ftrl_steps, num_states))
        # the values of past versions of d
        d_vals = np.empty((self.num_ftrl_steps, num_states, num_actions))

        num_samples = np.sum(
            [
                utils.num_samples(trajectories_over_t[a])
                for a in range(len(trajectories_over_t))
            ]
        )
        for step in range(self.num_ftrl_steps):

            """retrain h"""
            h = cp.Variable(num_states)
            constraints = []
            # ||h||_2 <= 3S/(1-\gamma)^2
            constraints.append(
                cp.pnorm(h, 2) <= 3 * num_states / cp.power((1 - self.gamma), 2)
            )

            h_linear_factors = np.zeros(num_states)

            if step == 0:
                d_val_lst = [initial_d]
            else:
                d_val_lst = d_vals[:step]

            for i_step in range(len(trajectories_over_t)):
                for trajectory in trajectories_over_t[i_step]:
                    for i_entry in range(len(trajectory) - 1):
                        for d_val in d_val_lst:
                            s = trajectory[i_entry][0]
                            next_s = trajectory[i_entry + 1][0]
                            a = trajectory[i_entry][1]
                            d_bar = occupancies_over_t[i_step]
                            h_linear_factors[s] += -(
                                d_val[s, a] / (d_bar[s, a] * (1 - self.gamma))
                            )
                            h_linear_factors[next_s] += (
                                self.gamma
                                * d_val[s, a]
                                / (d_bar[s, a] * (1 - self.gamma))
                            )
            h_linear_factors /= num_samples
            h_linear_factors += self._meta_env.start_distribution(policy_index)
            lagrangian_h = cp.scalar_product(h, h_linear_factors)

            # calculate the regularizing factor for h
            lagrangian_h += (
                self.regularize_h / 2 * cp.power(cp.pnorm(h, p=2), 2)  # type: ignore
            )

            # solve the minimization problem
            h_objective = cp.Minimize(lagrangian_h)
            problem = cp.Problem(objective=h_objective, constraints=constraints)
            try:
                problem.solve(**self.SOLVER_KWARGS)
                status = problem.status
                h_val = h.value
            except:
                status = "exception"
            # problem.solve(solver=cp.SCS, eps=tolerance_optimization)

            if status in ["infeasible", "unbounded", "exception"]:
                problem.solve(verbose=True, **self.SOLVER_KWARGS, canon_log=sys.stderr)
                # problem.solve(solver=cp.SCS, eps=tolerance_optimization,
                #              verbose=True)
                eprint("occupancies_over_t[0]", occupancies_over_t[0])
                eprint("lagrangian_h_linear_factors", h_linear_factors)
                eprint("d_val", d_val)  # type: ignore
                eprint("h_val", h_val)
                eprint("step", step)
                traceback.print_exc(file=sys.stderr)
                raise ValueError("h not found")

            """retrain d"""
            d = cp.Variable((num_states, num_actions), nonneg=True)
            # calculate the constraints
            # constraints = []
            # for occupancy in occupancies_over_t:
            #    constraints.append(self.B * occupancy >= d)
            # constraints.append(d>=np.)
            constraints = []
            for s in range(num_states):
                for a in range(num_actions):
                    constraints.append(d[s, a] >= 0)  # type: ignore
                    for occupancy in occupancies_over_t:
                        constraints.append(d[s, a] <= self.B * occupancy[s, a])

            # calculate the linear factors for d
            d_linear_factors = np.zeros((num_states, num_actions))
            for i_step in range(len(trajectories_over_t)):
                for trajectory in trajectories_over_t[i_step]:
                    for i_entry in range(len(trajectory) - 1):
                        s = trajectory[i_entry][0]
                        next_s = trajectory[i_entry + 1][0]
                        a = trajectory[i_entry][1]
                        r = trajectory[i_entry][2]
                        d_bar = occupancies_over_t[i_step]
                        d_linear_factors[s, a] += (
                            (r - h_val[s] + self.gamma * h_val[next_s])
                        ) / (d_bar[s, a] * (1 - self.gamma))
            d_linear_factors /= num_samples

            lagrangian_d = cp.scalar_product(d, d_linear_factors)

            # calculate the regularizing factor for d
            lagrangian_d -= self.lamda / 2 * cp.power(cp.norm(d, p="fro"), 2)  # type: ignore

            # solve the maximization problem
            d_objective = cp.Maximize(lagrangian_d)
            problem = cp.Problem(objective=d_objective, constraints=constraints)
            # problem.solve(solver=cp.SCS, eps=tolerance_optimization)
            try:
                problem.solve(**self.SOLVER_KWARGS)
                # problem.solve(solver=cp.ECOS, max_iters=100)
                d_val = d.value
                status = problem.status
            except:
                status = "exception"
            if (
                status
                in [
                    "infeasible",
                    "infeasible_inaccurate",
                    "unbounded",
                    "unbounded_inaccurate",
                    "exception",
                ]
                or d_val is None  # type: ignore
            ):
                problem.solve(verbose=True, **self.SOLVER_KWARGS, canon_log=sys.stderr)
                # problem.solve(solver=cp.SCS, eps=tolerance_optimization,
                #              verbose=True)
                # problem.solve(solver=cp.ECOS, max_iters=10000, verbose=True)
                eprint("occupancies_over_t[0]", occupancies_over_t[0])
                eprint("d_linear_factors", d_linear_factors)
                eprint(
                    "np.sum(np.abs(d_linear_factors))",
                    np.sum(np.abs(d_linear_factors)),
                )
                eprint("d_val", d_val)  # type: ignore
                eprint("h_val", h_val)
                eprint("step", step)
                eprint("problem.status", problem.status)

                raise ValueError("d not found")

            # add the current values of d and h to the list
            h_vals[step] = h_val
            d_vals[step] = d_val  # type: ignore

        # compute the mean of the d values and return this
        average_d = np.mean(d_vals, axis=0)
        return OccupancyMeasure(average_d)
