import socket

import click

from sperl.config.config_manager import ConfigManager
from sperl.experiment.experiment_manager import ExperimentManager
from sperl.metaenvs.girdworlds2agents.agent2_algorithm.decaying_mixture import (
    DecayingMixture,
)


@click.command()
@click.option("--seed", default=1234, type=int, help="A random seed.")
@click.option(
    "--beta",
    default=[1.0],
    type=float,
    multiple=True,
    help="A factor in the softmax",
)
@click.option(
    "--regularizer",
    multiple=True,
    default=[0.1],
    type=float,
    help="List of regularizing factors",
)
@click.option("--gamma", multiple=True, default=[0.9], type=float)
@click.option(
    "--num-trajectories",
    default=[100],
    type=int,
    multiple=True,
    help="The number of trajectories to use (per retraining).",
)
@click.option(
    "--num-ftrl-steps",
    default=[10],
    type=int,
    multiple=True,
    help="The number of ftrl steps",
)
@click.option(
    "--B",
    default=[10.0],
    type=float,
    multiple=True,
    help="The B parameter for the approximate optimization problem.",
)
@click.option(
    "--k",
    multiple=True,
    default=[3],
    type=int,
    help="Delays of DRR / MDRR (k in the algorithm)",
)
@click.option(
    "--agent2-algorithm",
    multiple=True,
    default=[
        (DecayingMixture.NAME, 0.5),
    ],
    nargs=2,
    type=click.Tuple([str, float]),
    help="value for agent two mixture (w)",
)
@click.option(
    "--meta-policy",
    multiple=True,
    default=["RR", "DRR", "MDRR"],
    type=click.Choice(["RR", "DRR", "MDRR"]),
    help="List of meta-policies to use.",
)
@click.option(
    "--max-trajectory-length",
    default=50,
    type=int,
)
@click.option("--agent2-gamma", default=0.9, type=float)
@click.option(
    "--num-deployments",
    type=int,
    default=100,
    help="The number of deployments",
)
@click.option(
    "--num-meta-trajectories",
    default=20,
    type=int,
    help="The number of meta-trajectories to use.",
)
@click.option(
    "--v",
    default=[1.1],
    multiple=True,
    type=float,
    help="Parameter v for the MDRR algorithm",
)
@click.option(
    "--previous-occupancy-mix",
    default=0.0,
    type=float,
    help="compute d_bar by using the current"
    "emperical occupancy and the last one (weighted with this value)",
)
@click.option("--exact-optimization", is_flag=True)
@click.option(
    "--base-path",
    default=f"./data_{socket.gethostname()}/",
    type=str,
    help="The base path for storage of figures and data.",
)
@click.option("--grid-world", default=0, type=int)
@click.option(
    "--num-jobs",
    default=48,
    type=int,
)
@click.option("--profile", is_flag=True)
@click.option(
    "--store-steps",
    default=1000,
    type=int,
)
def main(*args, **kwargs):
    if args:
        raise ValueError("Args should be empty.")
    config_manager = ConfigManager(**kwargs)
    experiment_manager = ExperimentManager(config_manager)
    experiment_manager.run()


if __name__ == "__main__":
    main()
