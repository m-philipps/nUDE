import os

job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

from models import pypesto_problems, nude_problems, TRAINING

model_id = list(nude_problems)[job_array_id]
pypesto_problem = pypesto_problems[TRAINING][model_id]

print(model_id)
