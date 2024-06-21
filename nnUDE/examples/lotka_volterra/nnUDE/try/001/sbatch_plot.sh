#!/bin/bash
#SBATCH --job-name nude_lv
#SBATCH --output log/%A_%a.txt
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH --time 7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dilan.pathirana@uni-bonn.de

source ../../../../../../environment/env.sh
#python calibrate.py
python plot.py
