Here's how to reproduce the Lotka-Volterra example from the paper.

1. From this directory, go to nnUDE/try/001.
2. We include our results inside the directory `output`. To re-run our work from scratch, delete this folder.
3. Run `python models.py` to generate the different model variations.
4. Run `python make_startpoints.py` to generate optimization start points that will be shared across all model variations.
5. Calibrate all models
  - if you run this on a cluster: customize/adapt `sbatch.sh` and `submit.sh` for your HPC and run `bash submit.sh`.
  - if you run this locally: edit `array_provider.py` and change `job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))` to `job_array_id = 0`. Then run `python calibrate.py`
    - This will calibrate the first model. Repeat for `job_array_id` in 0 to 6 to calibrate all 7 model variations.
  - Edit `cpus_per_task` in `calibrate.py` to adjust the number of CPUs that get utilized for calibration.
  - You will likely see many simulation errors -- this is expected. 1000 multi-starts are run, and some will fail. The number of multi-starts can be reduced to reduce the required compute time, but results may consequently worsen or be less reproducible.
6. Plot results
  - on a cluster, adjust `sbatch_plot.sh` and run `sbatch sbatch_plot.sh`.
  - locally, as for calibration, change `job_array_id` in `array_provider.py` and run `python plot.py` until all model variations are plotted.
  - edit `ThreadPoolExercutor(max_workers=48)` in `plot.py` to adjust the CPU utlization
7. All results (including plots) can be found in the `output` directory.
