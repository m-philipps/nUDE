import numpy as np
from typing import List, Dict


UNIVERSAL_MAGNITUDE = "nUDE_universal_magnitude"

UNIVERSAL_INTEGRAL = "nUDE_universal_integral"

MUTED = "nUDE_muted"

REGULARIZATION_STRENGTH = 'regularization_strength'


def setup_integral(model):
    universal_magnitude = " + ".join([f"({Ui}) * ({Ui})" for Ui in model.universal_rhs])
    universal_magnitude = f"sqrt({universal_magnitude})"

    model.sbml_handler.create_parameter_with_assignment(
        parameter_id=UNIVERSAL_MAGNITUDE,
        formula=universal_magnitude,
    )

    model.sbml_handler.create_parameter_with_rate(
        parameter_id=UNIVERSAL_INTEGRAL,
        formula=UNIVERSAL_MAGNITUDE,
    )


def add_integral_regularization(
    model,
    **kwargs,
):
    parameters = [p.getId() for p in model.sbml_handler.libsbml_model.getListOfParameters()]
    if UNIVERSAL_INTEGRAL not in parameters:
        setup_integral(model)

    add_regularization(
        model=model,
        parameter_id=UNIVERSAL_INTEGRAL,
        n_times=-1,
        **kwargs,
    )

    return model

def add_regularization(
    model,
    parameter_id: str,
    n_times: List[float] = None,
    strength: float = 1.0,
):

    observable_id = "observable__" + parameter_id
    model.petab_handler.add_observable(
        observable_id=observable_id,
        observable_formula=parameter_id,
        noise_formula=1/strength,
    )

    times = sorted(set(model.petab_handler.petab_problem.measurement_df.time))
    if n_times == -1:
        times = [times[-1]]
    elif n_times is not None:
        times = np.linspace(times[0], times[-1], n_times)

    model.petab_handler.add_measurements(
        observable_id=observable_id,
        times=times,
        measurements=[0.0] * len(times),
    )

def track_muted_dynamics(
    model,
    muted_dynamics: Dict[str, str],
):
    for state_variable_id, muted_dynamic in muted_dynamics.items():
        model.petab_handler.add_observable(
            observable_id=MUTED + "__" + state_variable_id,
            observable_formula=muted_dynamic,
            noise_formula='1',
        )

    other_tracked_components = {
        **{
            f'tracked_f_{i}': id_
            for i, id_ in enumerate(model.mechanistic_rhs)
        },
        **{
            f'tracked_U_{i}': id_
            for i, id_ in enumerate(model.universal_rhs)
        },
    }
    for tracking_id, tracked_id in other_tracked_components.items():
        model.petab_handler.add_observable(
            observable_id=tracking_id,
            observable_formula=tracked_id,
            noise_formula='1',
        )
