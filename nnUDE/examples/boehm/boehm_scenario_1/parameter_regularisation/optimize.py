import pypesto.optimize
import fides
import sys
from pathlib import Path
from hyperparameters import *

boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))
from helpers import *
from models import pypesto_importers, pypesto_problems

optimizer = pypesto.optimize.FidesOptimizer(
    verbose=0,
    hessian_update=fides.BFGS(),
    options={"maxiter": 10000},
)
engine = pypesto.engine.MultiProcessEngine()

for label in LABELS:
    pypesto_problem = pypesto_problems[label][REGULARIZATION_STRENGTH]
    output_path = Path(__file__).resolve().parent / "output" / GRID_ID
    output_path.mkdir(exist_ok=True, parents=True)
    result = pypesto.optimize.minimize(
        problem=pypesto_problem,
        optimizer=optimizer,
        n_starts=N_STARTS,
        engine=engine,
    )
    pypesto.store.write_result(
        result=result,
        filename=output_path / f"{label}_result_posterior.hdf5",
        optimize=True,
    )

    # Use unregularized objective to calculate likelihood instead of posterior
    update_fvals(
        fp_input=output_path / f"{label}_result_posterior.hdf5",
        fp_output=output_path / f"{label}_result_likelihood.hdf5",
        pypesto_problem=pypesto_problems[label][0],
    )
