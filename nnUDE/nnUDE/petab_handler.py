from typing import Any, List, Dict

import pandas as pd
import petab


from petab.C import (
    PARAMETER_ID,
    NOMINAL_VALUE,
    PARAMETER_SCALE,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
    INITIALIZATION_PRIOR_TYPE,
    INITIALIZATION_PRIOR_PARAMETERS,
    NORMAL,
    LIN,
    TIME,
    OBSERVABLE_ID,
    MEASUREMENT,
    DATASET_ID,
    SIMULATION_CONDITION_ID,
    OBSERVABLE_FORMULA,
    NOISE_FORMULA,
    UNIFORM,
)


from .sbml_handler import SbmlHandler
from .neural_network import Node



class PetabHandler:
    def __init__(
        self,
        petab_problem: petab.Problem,
    ):
        self.petab_problem = petab_problem


    def add_parameter(
        self,
        data: Dict[str, Any],
    ):
        new_parameter_df = pd.DataFrame(
            data=[list(data.values())],
            columns=list(data.keys()),
        )
        new_parameter_df = petab.get_parameter_df(new_parameter_df)
    
        self.petab_problem.parameter_df = pd.concat(
            [
                self.petab_problem.parameter_df,
                new_parameter_df,
            ],
            verify_integrity=True,
        )

    def add_measurements(
        self,
        observable_id: str,
        times: List[float],
        measurements: List[float],
    ):
        if len(times) != len(measurements):
            raise ValueError

        data = [
            {
                OBSERVABLE_ID: observable_id,
                SIMULATION_CONDITION_ID: 'condition1',
                TIME: time,
                MEASUREMENT: measurement,
                DATASET_ID: 'regularization',
            }
            for time, measurement in zip(times, measurements)
        ]

        new_measurement_df = pd.DataFrame(
            data=data,
            columns=list(data[0].keys()),
        )
    
        self.petab_problem.measurement_df = petab.get_measurement_df(pd.concat(
            [
                self.petab_problem.measurement_df,
                new_measurement_df,
            ],
            #verify_integrity=True,
        ))
    
    def add_observable(
        self,
        observable_id: str,
        observable_formula: str,
        noise_formula: str,
    ):
        if observable_id in self.petab_problem.observable_df.index:
            raise ValueError(
                f"Cannot add observable with ID `{observable_id}` to the "
                "PEtab observable table. An observable with that ID already "
                "exists."
            )

        self.petab_problem.observable_df.loc[observable_id] = {
            OBSERVABLE_FORMULA: observable_formula,
            NOISE_FORMULA: noise_formula,
        }


def enable_small_startpoints(
    petab_problem: petab.Problem,
    parameter_ids: List[str],
    multiplier: float = 1.5,
):
    for parameter_id in parameter_ids:
        petab_problem.parameter_df.loc[parameter_id, INITIALIZATION_PRIOR_TYPE] = UNIFORM
        lb = petab_problem.parameter_df.loc[parameter_id, LOWER_BOUND]
        ub = petab_problem.parameter_df.loc[parameter_id, UPPER_BOUND]
        if ub < lb*multiplier:
            raise ValueError(f"Upper bound too low, decrease small startpoints multiplier. Max multiplier for parameter {parameter_id}: {ub/lb}")
        petab_problem.parameter_df.loc[parameter_id, INITIALIZATION_PRIOR_PARAMETERS] = f"{lb};{lb*multiplier}"

