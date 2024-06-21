"""pypesto importers and problems for non-regularized setup and dfferent regularisation strengths."""

import nnUDE
import pypesto.petab
from hyperparameters import *

import sys
from pathlib import Path

boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))
from helpers import *

pypesto_importers = {
    UDE: {},
    NUDE: {},
    BNUDE: {},
}
pypesto_problems = {
    UDE: {},
    NUDE: {},
    BNUDE: {},
}

nn_kwargs = {
    UDE: {
        "non_negative": False,
        "non_negative_bounded": False,
    },
    NUDE: {
        "non_negative": True,
        "non_negative_bounded": False,
    },
    BNUDE: {
        "non_negative": False,
        "non_negative_bounded": True,
    },
}


def get_neural_network():
    return nnUDE.create_feedforward_network(
        neural_network_id="nn1",
        dimensions=NN_DIMENSIONS,
        activation_function=ACTIVATION_FUNCTION,
        input_species_ids=NN_INPUT_IDS,
        output_species_ids=NN_OUTPUT_IDS,
    )


def get_startpoints():
    return np.loadtxt(
        Path(__file__).resolve().parent / "output" / "startpoints.tsv", delimiter="\t"
    )


for label in LABELS:
    for reg in list(set([0, REGULARIZATION_STRENGTH])):
        petab_problem = get_muted_petab_problem()

        nnude_model = nnUDE.Model(petab_problem=petab_problem)
        nnude_model.add_neural_network(
            neural_network=get_neural_network(),
            **nn_kwargs[label],
        )

        if reg:
            regularize_petab_problem_with_l2(
                nnude_model.petab_handler.petab_problem, reg
            )

        model_name = f"boehm_{label}"
        pypesto_importer = pypesto.petab.PetabImporter(
            petab_problem=nnude_model.petab_handler.petab_problem,
            model_name=model_name,
            output_folder=Path(__file__).resolve().parent / "amici_models" / model_name,
        )
        pypesto_objective = pypesto_importer.create_objective()
        fix_objective(pypesto_objective)
        pypesto_problem = pypesto_importer.create_problem(objective=pypesto_objective)

        # set startpoints, if available
        try:
            pypesto_problem.set_x_guesses(
                pypesto_problem.get_full_vector(get_startpoints())
            )
        except FileNotFoundError:
            pass

        pypesto_importers[label][reg] = pypesto_importer
        pypesto_problems[label][reg] = pypesto_problem
