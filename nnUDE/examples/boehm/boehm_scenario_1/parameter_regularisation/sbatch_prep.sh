#!/bin/bash
#SBATCH --job-name boehm1_preg_prep
#SBATCH --output log/%j.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 01-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maren.philipps@uni-bonn.de

source /home/maren/zenodo2/environment/env.sh
python make_startpoints.py

