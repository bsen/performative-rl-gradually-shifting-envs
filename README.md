# Code for "Performative Reinforcement Learning in Gradually Shifting Environments"

### Notes on the plots
All generated plots show $\pm 1.96 \cdot \text{standard error}$ interval and the mean.

### Prerequisites
- operating system: `linux`
- `conda` (available [here](https://www.anaconda.com/download/))
- install the environment for this repository via `conda env create --name sperl --file=sperl_environment.yml`
- activate the environment via `conda activate sperl`

### Run experiments
#### $w=0.5$
##### Generate data
First run (those commands can be run concurrently)
```
bash scripts/for_submission/w.5/drr_k3.sh
bash scripts/for_submission/w.5/mdrr_k3.sh
bash scripts/for_submission/w.5/rr.sh
```

##### Generate plots from data
After the above scripts finished, run
```
bash scripts/for_submission/generate_plots/w.5.sh
```
The figures are to be found in `data/w.5`.

#### $w=0.15$
##### Generate data
First run (those commands can be run concurrently)
```
bash scripts/for_submission/w.15/drr_k3.sh
bash scripts/for_submission/w.15/mdrr_k3.sh
bash scripts/for_submission/w.15/rr.sh
```

##### Generate plots from data
After the above scripts finished, run
```
bash scripts/for_submission/generate_plots/w.15.sh
```
The figures are to be found in `data/w.15`.

#### Compare values of $v$
##### Generate data
First run (those commands can be run concurrently)
```
bash scripts/for_submission/vs/mdrr_v1.1.sh
bash scripts/for_submission/vs/mdrr_v1.5.sh
bash scripts/for_submission/vs/mdrr_v1.8.sh
```


##### Generate plot from data
After the above scripts finished, run
```
bash scripts/for_submission/generate_plots/vs.sh
```
The figure is to be found in `data/vs`.

#### Compare values of $k$
##### Generate data
First run (those commands can be run concurrently)
```
bash scripts/for_submission/w.5/mdrr_k10.sh
```
If the commands for comparing values of $v$ were not run yet, please also run
```
bash scripts/for_submission/vs/mdrr_v1.1.sh
```
If the commands for $w=0.5$ were not run yet, please also run
```
bash scripts/for_submission/w.5/mdrr_k3.sh
```

##### Generate plot from data
After the above scripts finished, run
```
bash scripts/for_submission/generate_plots/ks.sh
```
The figure is to be found in `data/compare_ks`.

### Rerunning Data Generation
To remove artifacts if you want to rerun certain commands in the data generation. (E.g. if you cancelled the script in the first attempt.)

**Only use this commands in the folder of this repository, nowhere else.**

For `bash scripts/for_submission/w.5/drr_k3.sh`
```
rm -r data/w.5/drr_k3
```
For `bash scripts/for_submission/w.5/mdrr_k3.sh`
```
rm -r data/w.5/mdrr_k3
```
For `bash scripts/for_submission/w.5/rr.sh`
```
rm -r data/w.5/rr
```
For rerunning `bash scripts/for_submission/w.15/drr_k3.sh`
```
rm -r data/w.15/drr_k3
```
For rerunning `bash scripts/for_submission/w.15/mdrr_k3.sh`
```
rm -r data/w.15/mdrr_k3
```
For rerunning `bash scripts/for_submission/w.15/rr.sh`
```
rm -r data/w.15/rr
```
For rerunning `bash scripts/for_submission/vs/mdrr_v1.1.sh`
```
rm -r data/vs/mdrr_v1.1
```
For rerunning `bash scripts/for_submission/vs/mdrr_v1.5.sh`
```
rm -r data/vs/mdrr_v1.5
```
For rerunning `bash scripts/for_submission/vs/mdrr_v1.8.sh`
```
rm -r data/vs/mdrr_v1.8
```
For rerunning `bash scripts/for_submission/w.5/mdrr_k10.sh`
```
rm -r data/w.5/mdrr_k10
```
