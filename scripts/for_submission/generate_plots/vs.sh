#!/bin/bash
folder="data/vs/"
i1="mdrr_v1.1/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=5_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"
i2="mdrr_v1.5/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=5_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.5.pickle"
i3="mdrr_v1.8/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=5_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.8.pickle"
compare_variable="v"
output_name='vs'

python -m sperl.plotter.plotter_main --distance occupancy-last --x-axis step --compare-variable "$compare_variable" \
"$@" --folder "$folder" --input-file "$i1" --input-file "$i2" --input-file "$i3" --output-file "$output_name" --distance-range 0 6 --canvas 3.3 3.3 --colors-alt 1
