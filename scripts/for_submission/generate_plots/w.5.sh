#!/bin/bash
folder="data/w.5/"
i1="rr/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=RR_v=1.1.pickle"
i2="drr_k3/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=DRR_v=1.1.pickle"
i3="mdrr_k3/exp_000/steps=12000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"

compare_variable="meta_policy"
output_name='meta_pol'

python -m sperl.plotter.plotter_main "$@" --distance occupancy-last --x-axis step --compare-variable "$compare_variable"  \
--folder "$folder" --input-file "$i1" --input-file "$i2" --input-file "$i3" --output-file "$output_name" --distance-range 0 6 --canvas 3 3.3

# reward comparison
i1="rr/exp_000/steps=6000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=RR_v=1.1.pickle"
i2="drr_k3/exp_000/steps=6000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=DRR_v=1.1.pickle"
i3="mdrr_k3/exp_000/steps=6000_beta=1.0_regularizer=0.1_gamma=0.9_num_trajectories=1000_num_ftrl_steps=10_b=10.0_k=3_agent2_algorithm=('decaying-mixture', 0.5)_meta_policy=MDRR_v=1.1.pickle"
python -m sperl.plotter.plotter_main --distance reward --x-axis step --compare-variable "$compare_variable"  --canvas 6 3.45 \
"$@" --folder "$folder" --input-file "$i1" --input-file "$i2" --input-file "$i3" --output-file "$output_name" --distance-range -0.37 -0.23 \
--fontsize-axis 24 --fontsize-numbers 21 --fontsize-legend 16 --thicker-axis 
