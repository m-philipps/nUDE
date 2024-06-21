import os

from array_provider import pypesto_problem, model_id

from pathlib import Path
import pypesto.engine
import pypesto.optimize
import pypesto.store
import fides

cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
if cpus_per_task == 0:
    # let pyPESTO decide
    cpus_per_task = None

def calibrate(pypesto_problem):
    n_starts = 1000

    if pypesto_problem.x_guesses.shape[0] != n_starts:
        raise ValueError("please specify startpoints")

    pypesto_optimizer = pypesto.optimize.FidesOptimizer(
        verbose=0,
        hessian_update=fides.BFGS(),
        options={'maxiter': 10000},
    )
    pypesto_engine = pypesto.engine.MultiProcessEngine(cpus_per_task)
    
    pypesto_result = pypesto.optimize.minimize(
        problem=pypesto_problem,
        optimizer=pypesto_optimizer,
        n_starts=n_starts,
        engine=pypesto_engine,
    )
    return pypesto_result


def save_result(model_id, pypesto_result):
    output_path = Path(__file__).resolve().parent / "output" / "result" / f"{model_id}.hdf5"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    pypesto.store.write_result(
        result=pypesto_result,
        filename=output_path,
        problem=False,
        optimize=True,
    )

pypesto_result = calibrate(pypesto_problem=pypesto_problem)
save_result(model_id=model_id, pypesto_result=pypesto_result)
