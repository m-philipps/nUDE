import os
import nnUDE

slurm_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", False))

N_STARTS = 100

species_ids = [
    "STAT5A",
    "STAT5B",
    "pApB",
    "pApA",
    "pBpB",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]
NN_INPUT_IDS = [
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]
NN_OUTPUT_IDS = [
    "STAT5A",
    "STAT5B",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]

ACTIVATION_FUNCTION = nnUDE.ActivationFunction.tanh
NN_DIMENSIONS = [5] + [5] + [5] + [len(NN_OUTPUT_IDS)]

# hp grid
REGULARIZATION_STRENGTHS = (
    int(0),
#     1e-2,
    1e-1,
#     int(1e0),
    int(1e1),
#     int(1e2),
)
REGULARIZATION_STRENGTH = REGULARIZATION_STRENGTHS[slurm_id]

GRID_ID = "reg_" + str(REGULARIZATION_STRENGTH)
