from typing import Optional
import click

from sperl.config.experiment import ExperimentConfig
from sperl.generatedata.measures.all_measures import AllMeasures
from sperl.plotter.manager import PlotterManager

COMPARE_VARIABLES = list(ExperimentConfig().__dict__.keys())
# TESTING = True
TESTING = False


@click.command()
@click.option(
    "--distance",
    type=click.Choice([d.name() for d in AllMeasures.DISTANCES]),
    default="occupancy",
)  # type: ignore
@click.option(
    "--x-axis",
    type=click.Choice(["step", "samples", "retrainings"]),
    default="step",
)
@click.option(
    "--compare-variable",
    type=click.Choice(COMPARE_VARIABLES),
    default="meta_policy",
)
@click.option("--folder", type=str, required=not TESTING)
@click.option("--input-file", multiple=True, required=not TESTING)
@click.option("--logscale", is_flag=True)
@click.option(
    "--distance-range",
    required=False,
    default=None,
    type=click.Tuple([float, float]),
)
@click.option("--output-file", type=str, default="")
@click.option("--resource-min", required=False, default=0, type=int)
@click.option(
    "--resource-max", required=False, default=PlotterManager.LARGE_INT, type=int
)
@click.option(
    "--canvas",
    required=False,
    default=(6.0, 5.0),
    type=click.Tuple([float, float]),
)
@click.option("--fontsize-axis", default=18, type=int, required=False)
@click.option("--fontsize-numbers", default=16, type=int, required=False)
@click.option("--fontsize-legend", default=16, type=int, required=False)
@click.option("--thicker-axis", is_flag=True)
@click.option("--print-resource-value", default=-100.0)
@click.option('--overlay-speedup', default=None, type=str)
@click.option('--colors-alt', type=int, default=0)
def main(*args, **kwargs):
    assert len(args) == 0
    if TESTING:
        kwargs["distance"] = "reward"
        kwargs["x_axis"] = "retrainings"
        kwargs["compare_variable"] = "meta_policy"
        kwargs["folder"] = "data/gw5_decaying_deployments_testing/"
        kwargs["input_file"] = [
            "experiment_custom_drr_k10/exp_002/beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=10_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=DRR_v=1.1.pickle",
            "experiment_custom_mdrr_k10/exp_002/beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=10_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle",
            "experiment_custom_rr/exp_003/beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=RR_v=1.1.pickle",
        ]
        kwargs["output_file"] = None
    m = PlotterManager(**kwargs)
    m.run()


if __name__ == "__main__":
    main()
