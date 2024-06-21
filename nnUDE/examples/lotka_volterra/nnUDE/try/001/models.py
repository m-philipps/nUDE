#SMALL_STARTPOINTS = True

from pathlib import Path
import copy
import nnUDE
import pandas as pd
import petab
from petab.C import (
    TIME,
    DATASET_ID,
)
from nnUDE.regularization import (
    add_integral_regularization,
    track_muted_dynamics,
)
import amici
import pypesto.petab

#petab_yaml = "../../../petab/petab_problem.yaml"
petab_yaml = "/home/dilan/projects/nnude/github/nnUDE/examples/lotka_volterra/petab/petab_problem.yaml"
data_tsv = "/home/dilan/projects/nnude/github/nnUDE/examples/lotka_volterra/nnUDE/data/dataset.tsv"

from _helpers import startpoints  #, startpoints_small


def fix_objective(objective):
    objective.amici_solver.setMaxSteps(int(1e5))
    objective.amici_solver.setAbsoluteTolerance(1e-8)
    objective.amici_solver.setRelativeTolerance(1e-6)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)

#if SMALL_STARTPOINTS:
#    startpoints = startpoints_small

TRAINING = 'training'
TRAINING_UNREGULARIZED = 'training_unregularized'
PREDICT = 'predict'

STAGES = [
    TRAINING,
    TRAINING_UNREGULARIZED,
    PREDICT,
]


def mute_problem(model):
    """Mute some mechanisms in the ODEs, which the nnUDEs should recover."""
    petab_problem = model.petab_handler.petab_problem
    for deleted_parameter_id in ['beta_', 'delta_']:
        petab_problem.parameter_df.loc[deleted_parameter_id, petab.C.ESTIMATE] = 0
        petab_problem.parameter_df.loc[deleted_parameter_id, petab.C.NOMINAL_VALUE] = 0

    add_tracked_muted_dynamics(model)
    return model


def add_tracked_muted_dynamics(model):
    muted_dynamics = {
        "prey": "beta_ * prey * predator",
        "predator": "delta_ * prey * predator",
    }

    track_muted_dynamics(
        model=model,
        muted_dynamics=muted_dynamics,
    )


def get_neural_network():
    neural_network_id = "nn1"
    dimensions = [2,2]
    activation_function = nnUDE.ActivationFunction.tanh
    species_ids = [
        "predator",
        "prey",
    ]

    neural_network = nnUDE.create_feedforward_network(
        neural_network_id=neural_network_id,
        dimensions=dimensions,
        activation_function=activation_function,
        input_species_ids=species_ids,
        output_species_ids=species_ids,
        identity_last_layer=True,
    )
    return neural_network


def ude_problem(ude_model):
    ude_model.add_neural_network(
        neural_network=get_neural_network(),
        non_negative=False,
        non_negative_bounded=False,
    )

    return ude_model


def nude_problem(nude_model):
    nude_model.add_neural_network(
        neural_network=get_neural_network(),
        non_negative=True,
        non_negative_bounded=False,
    )

    return nude_model


def bounded_nude_problem(nude_model):
    nude_model.add_neural_network(
        neural_network=get_neural_network(),
        non_negative=False,
        non_negative_bounded=True,
        non_negative_prefactor=1,
    )

    return nude_model


def sharply_bounded_nude_problem(nude_model):
    nude_model.add_neural_network(
        neural_network=get_neural_network(),
        non_negative=False,
        non_negative_bounded=True,
        non_negative_prefactor=10,
    )

    return nude_model


# Expected ODE
expected_ode_petab_problem = petab.Problem.from_yaml(petab_yaml)
expected_ode_petab_problem.measurement_df = petab.get_measurement_df(data_tsv)
expected_ode_pypesto_importer = pypesto.petab.PetabImporter(
    petab_problem=expected_ode_petab_problem,
    output_folder=f"amici_models/expected_ode2",
    model_name="expected_ode",
)
expected_ode_objective = expected_ode_pypesto_importer.create_objective()
fix_objective(expected_ode_objective)
expected_ode_pypesto_problem = expected_ode_pypesto_importer.create_problem(
    expected_ode_objective
)
expected_ode_result = expected_ode_pypesto_problem.objective(
    expected_ode_petab_problem.x_nominal_free_scaled,
    return_dict=True,
)

# ODE
ode_petab_problem = petab.Problem.from_yaml(petab_yaml)


def get_model():
    petab_problem = petab.Problem.from_yaml(petab_yaml)
    training_data = petab.get_measurement_df("../../data/dataset.tsv")
    training_data = training_data.loc[training_data[TIME] <= 20]
    petab_problem.measurement_df = training_data
    return nnUDE.Model(petab_problem=petab_problem)


ude__muted__problem = (
    ude_problem(
    mute_problem(
        get_model()
    )
    )
)

nude__muted__problem = (
    nude_problem(
    mute_problem(
        get_model()
    )
    )
)

bounded__nude__muted__problem = (
    bounded_nude_problem(
    mute_problem(
        get_model()
    )
    )
)

sharply_bounded__nude__muted__problem = (
    sharply_bounded_nude_problem(
    mute_problem(
        get_model()
    )
    )
)

REGULARIZATION_STRENGTH = 0.01

integral_regularized__nude__muted__problem = (
    add_integral_regularization(
        nude_problem(
            mute_problem(
                get_model()
            )
        ),
        strength=REGULARIZATION_STRENGTH,
    )
)

bounded__integral_regularized__nude__muted__problem = (
    add_integral_regularization(
        bounded_nude_problem(
        mute_problem(
            get_model()
        )
        ),
        strength=REGULARIZATION_STRENGTH,
    )
)

sharply_bounded__integral_regularized__nude__muted__problem = (
    add_integral_regularization(
        sharply_bounded_nude_problem(
        mute_problem(
            get_model()
        )
        ),
        strength=REGULARIZATION_STRENGTH,
    )
)


nude_problems = {
    "ude": ude__muted__problem,
    "nude": nude__muted__problem,
    "integral_regularized_nude": integral_regularized__nude__muted__problem,
    "bounded_nude": bounded__nude__muted__problem,
    "sharply_bounded_nude": sharply_bounded__nude__muted__problem,
    "bounded_integral_regularized_nude": bounded__integral_regularized__nude__muted__problem,
    "sharply_bounded_integral_regularized_nude": sharply_bounded__integral_regularized__nude__muted__problem,
}

model_names = {
    'ude': 'UDE',
    'nude': 'nUDE\n$N(x)=x$',
    'integral_regularized_nude': 'nUDE\n$N(x)=x$\n$\lambda_o = 0.01$',
    'bounded_nude': 'nUDE\n$N(x)=\\tanh(x)$',
    'sharply_bounded_nude': 'nUDE\n$N(x)=\\tanh(10x)$',
    'bounded_integral_regularized_nude': 'nUDE\n$N(x)=\\tanh(x)$\n$\lambda_o = 0.01$',
    'sharply_bounded_integral_regularized_nude': 'nUDE\n$N(x)=\\tanh(10x)$\n$\lambda_o = 0.01$',
}

petab_problems = {}

petab_problems[TRAINING] = {
    problem_id: problem.petab_handler.petab_problem
    for problem_id, problem in nude_problems.items()
}





def get_training_data():
    training_data = petab.get_measurement_df("../../data/dataset.tsv")
    training_data = training_data.loc[training_data[TIME] <= 20]
    return training_data


if True:
    petab_problems[PREDICT] = {
        model_id: copy.deepcopy(petab_problem)
        for model_id, petab_problem in petab_problems[TRAINING].items()
    }

    petab_problems[TRAINING_UNREGULARIZED] = {
        model_id: copy.deepcopy(petab_problem)
        for model_id, petab_problem in petab_problems[TRAINING].items()
    }
    
    for _, petab_problem in petab_problems[TRAINING].items():
        if DATASET_ID in petab_problem.measurement_df.columns:
            regularization_data = petab_problem.measurement_df.loc[
                petab_problem.measurement_df[DATASET_ID] == "regularization"
            ]
        else:
            regularization_data = pd.DataFrame(data={})

        training_data = get_training_data()
        petab_problem.measurement_df = pd.concat([training_data, regularization_data])
    
    for _, petab_problem in petab_problems[TRAINING_UNREGULARIZED].items():
        training_data = get_training_data()
        petab_problem.measurement_df = training_data
    
    for _, petab_problem in petab_problems[PREDICT].items():
        petab_problem.measurement_df = petab.get_measurement_df("../../data/dataset.tsv")
    
    pypesto_importers = {}
    pypesto_problems = {}
    pypesto_problems = {
        stage: {}
        for stage in STAGES
    }
    for model_id in nude_problems:
        for stage in STAGES:
            petab_problem = petab_problems[stage][model_id]

            petab_path = Path(__file__).resolve().parent / "output" / "petab" / model_id / stage
            petab_path.mkdir(exist_ok=True, parents=True)
            petab_problem.to_files_generic(petab_path)
    
            pypesto_importer = pypesto.petab.PetabImporter(
                petab_problem=petab_problem,
                output_folder=f"amici_models/{model_id}",
                model_name=model_id,
            )
    
            pypesto_objective = pypesto_importer.create_objective()
            fix_objective(pypesto_objective)
    
            pypesto_problem = pypesto_importer.create_problem(objective=pypesto_objective)
            if startpoints is not None:
                pypesto_problem.set_x_guesses(pypesto_problem.get_full_vector(startpoints))
            
            pypesto_problems[stage][model_id] = pypesto_problem
            pypesto_importers[model_id] = pypesto_importer

