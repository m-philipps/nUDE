#!/bin/bash
#SBATCH --job-name boehm2_preg_prep
#SBATCH --output log/%j.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maren.philipps@uni-bonn.de

source /home/maren/zenodo2/environment/env.sh
python make_startpoints.py
