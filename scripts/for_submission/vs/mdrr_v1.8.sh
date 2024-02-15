#!/bin/bash
last_directory=$(basename $(dirname "$0"))
script_name=$(basename "$0" .sh)
data_dir="data/${last_directory}/${script_name}"
python -m sperl.experiment.experiment_main "$@" --num-meta-trajectories 20 --num-deployments 12003 --grid-world 5 --num-jobs 48 --num-trajectories 1000 --num-ftrl-steps 10 \
--agent2-algorithm decaying-mixture 0.5 \
--k 5 --beta 1 --meta-policy MDRR \
--base-path "$data_dir" --v 1.8
