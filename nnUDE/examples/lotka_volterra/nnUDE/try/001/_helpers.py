from pathlib import Path
import numpy as np

startpoints_path = Path(__file__).resolve().parent / "output" / "startpoints.tsv"
startpoints_path.parent.mkdir(exist_ok=True, parents=True)

#startpoints_small_path = Path(__file__).resolve().parent / "output" / "startpoints_small.tsv"
#startpoints_small_path.parent.mkdir(exist_ok=True, parents=True)


startpoints = None
if startpoints_path.exists():
    startpoints = np.loadtxt(startpoints_path, delimiter='\t')

#startpoints_small = None
#if startpoints_small_path.exists():
#    startpoints_small = np.loadtxt(startpoints_small_path, delimiter='\t')
