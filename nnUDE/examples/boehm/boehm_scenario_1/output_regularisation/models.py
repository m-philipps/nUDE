"""pypesto importers and problems for non-regularized setup and dfferent regularisation strengths."""

from typing import Tuple
import nnUDE
from nnUDE.regularization import add_integral_regularization
import pypesto.petab
from hyperparameters import *

import sys
from pathlib import Path
boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))
from helpers import *

pypesto_importers = {
    UDE: None,
    NUDE: None,
    BNUDE: None,
}
pypesto_problems = {
    UDE: None,
    NUDE: None,
    BNUDE: None,
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

def get_problem_by_setting(
    label: str,
    regularisation: float,
    **nn_kwargs,
) -> Tuple[pypesto.petab.PetabImporter, pypesto.Problem]:
    amici_model_id = f"boehm_{label}"
    if regularisation:
        amici_model_id += "_intreg_" + str(regularisation).replace(".", "")
        if NORMALISE_INTEGRAL:
            amici_model_id += "_normalised"

    nnude_model = nnUDE.Model(
        petab_problem=get_muted_petab_problem(),
    )
    nnude_model.add_neural_network(
        neural_network=get_neural_network(),
        **nn_kwargs,
    )

    if regularisation:
        nnude_model = add_integral_regularization(
            model=nnude_model,
            strength=regularisation,
        )
        nnude_model.petab_handler.petab_problem.measurement_df[
            SIMULATION_CONDITION_ID
        ] = "model1_data1"

    pypesto_importer = pypesto.petab.PetabImporter(
        petab_problem=nnude_model.petab_handler.petab_problem,
        output_folder=f"amici_models/{amici_model_id}",
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

    return pypesto_importer, pypesto_problem


def get_unregularised_problem(
    label: str,
    **nn_kwargs,
) -> Tuple[pypesto.petab.PetabImporter, pypesto.Problem]:
    return get_problem_by_setting(label, 0, **nn_kwargs)


def get_unregularised_problems() -> Tuple[
    dict[str : pypesto.petab.PetabImporter],
    dict[str : pypesto.Problem],
]:
    pypesto_importers_nonreg = {
        UDE: None,
        NUDE: None,
        BNUDE: None,
    }
    pypesto_problems_nonreg = {
        UDE: None,
        NUDE: None,
        BNUDE: None,
    }
    for label in LABELS:
        pypesto_importer, pypesto_problem = get_unregularised_problem(
            label, **nn_kwargs[label]
        )
        pypesto_importers_nonreg[label] = pypesto_importer
        pypesto_problems_nonreg[label] = pypesto_problem
    return pypesto_importers_nonreg, pypesto_problems_nonreg


for label in LABELS:
    pypesto_importer, pypesto_problem = get_problem_by_setting(
        label, REGULARIZATION_STRENGTH, **nn_kwargs[label]
    )
    pypesto_importers[label] = pypesto_importer
    pypesto_problems[label] = pypesto_problem

