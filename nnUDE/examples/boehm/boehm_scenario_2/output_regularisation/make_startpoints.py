import sys
from pathlib import Path
boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))

from helpers import *
from hyperparameters import *
import numpy as np
from models import pypesto_importers, pypesto_problems

seed = 2

output_path = Path(__file__).resolve().parent / "output"
output_path.mkdir(exist_ok=True)
startpoints_path = output_path / "startpoints.tsv"

pypesto_importer = pypesto_importers[UDE]
pypesto_problem = pypesto_problems[UDE]


def startpoint_method_petab(n_starts: int, **kwargs):
    return petab.sample_parameter_startpoints(
        pypesto_importer.petab_problem.parameter_df, n_starts=n_starts, seed=seed
    )


startpoint_method_pypesto = pypesto.startpoint.FunctionStartpoints(
    function=startpoint_method_petab
)
startpoints = startpoint_method_pypesto(
    n_starts=N_STARTS,
    problem=pypesto_problem,
)
# save startpoints
np.savetxt(startpoints_path, startpoints, delimiter="\t")

