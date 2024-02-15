#!/bin/bash
folder="data/compare_ks"
i1="../w.5/mdrr_k3/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"
i2="../vs/mdrr_v1.1/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=5_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"
i3="../w.5/mdrr_k10/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=10_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"
compare_variable="k"
output_name='compare_ks'

python -m sperl.plotter.plotter_main --distance occupancy-last --x-axis step --compare-variable "$compare_variable" \
"$@" --folder "$folder" --input-file "$i1" --input-file "$i2" --input-file "$i3" --output-file "$output_name" --distance-range 0 6 --canvas 3.3 3.3 --colors-alt 1
