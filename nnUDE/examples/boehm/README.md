## Workflow for reproducing the results

There are four folders:
- `boehm_scenario_1/output_regularisation`
- `boehm_scenario_1/parameter_regularisation`
- `boehm_scenario_2/output_regularisation`
- `boehm_scenario_2/parameter_regularisation`

To reproduce results from scratch for a particular Scenario and regularisation combination, go to the corresponding folder and:

1. Activate the virtual environment with the installed `nnUDE` package
2. Run `python make_startpoints.py` to generate optimization start points and precompile the amici models
3. Calibrate all models using `optimize.py`. Repeat for every regularisation setting.
4. Evaluate and visualise results for each experiment with the `evaluate.py` script. Repeat for every regularisation setting.
6. `evaluate_regularisation.py`: Compare results between regularisation strengths

If you run this locally:
- Repeat calibration and evaluation for every regularisation strength by modifying `hyperparameters.py`: Set the `slurm_id` to the integer that corresponds to the index in `REGULARIZATION_STRENGTHS`.

If you run this on a cluster, customise/adapt `submit.sh`, `sbatch_prep.sh`, `sbatch_expriments.sh` for your HPC
- `submit.sh` will trigger `sbatch_prep` and execute steps 1-2
- Run `sbatch sbatch_expriments.sh` to repeat steps 3 and 4 for all regularisation strenghts set in `hyperparameters.py`

### Notes
  - Edit `hyperparameters.py` to adjust e.g. the number of starts that will be run per model.
  - You will likely see many simulation errors -- this is expected. 100 multi-starts are run, and some will fail. The number of multi-starts can be reduced to reduce the required compute time, but results may consequently worsen or be less reproducible.

