#!/bin/bash
#SBATCH --job-name boehm1_oreg
#SBATCH --output log/%j.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --time 07-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maren.philipps@uni-bonn.de
#SBATCH --array=0-5%1

source /home/maren/zenodo2/environment/env.sh
python optimize.py
python evaluate.py
