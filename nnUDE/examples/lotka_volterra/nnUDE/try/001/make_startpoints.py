from models import pypesto_problems, pypesto_importers, TRAINING  #, SMALL_STARTPOINTS
from _helpers import startpoints_path  #, startpoints_small_path

import numpy as np


startpoint_method = pypesto_importers["nude"].create_startpoint_method()

nnude_startpoints = startpoint_method(
    n_starts=1000,
    problem=pypesto_problems[TRAINING]["nude"],
)
#np.savetxt(startpoints_small_path if SMALL_STARTPOINTS else startpoints_path, nnude_startpoints, delimiter='\t')
np.savetxt(startpoints_path, nnude_startpoints, delimiter='\t')
