import petab
petab_problem = petab.Problem.from_yaml('../petab/petab_problem.yaml')

import copy
import numpy as np
import amici.petab_simulate

from petab.C import TIME, MEASUREMENT

# Generate synthetic data
dummy_prey_data, dummy_predator_data = petab_problem.measurement_df.loc[0], petab_problem.measurement_df.loc[400]
all_data = []
for time in np.linspace(0, 100, 1001):
    prey_data = copy.deepcopy(dummy_prey_data)
    predator_data = copy.deepcopy(dummy_predator_data)
    prey_data["time"] = time
    predator_data["time"] = time
    all_data.append(prey_data)
    all_data.append(predator_data)
import pandas as pd

dummy_df = pd.DataFrame(data=all_data)
dummy_df = dummy_df.reset_index(drop=True)

petab_problem.measurement_df = dummy_df


simulator = amici.petab_simulate.PetabSimulator(petab_problem=petab_problem)
synthetic_df = simulator.simulate()

rng = np.random.default_rng(seed=0)
noise_level = 0.15  # `0.15` is data with 15% (multiplicative) noise
synthetic_noisy_df = copy.deepcopy(synthetic_df)
synthetic_noisy_df[MEASUREMENT] = rng.normal(loc=synthetic_df[MEASUREMENT], scale=noise_level * synthetic_df[MEASUREMENT])
#synthetic_noisy_df = synthetic_noisy_df.loc[synthetic_noisy_df[TIME] <= 20]

import petab.visualize
import matplotlib.pyplot as plt

petab_problem.measurement_df = synthetic_df
petab.visualize.plot_problem(
    petab_problem=petab_problem,
)
plt.savefig("validation_dataset.png")
petab.write_measurement_df(df=petab_problem.measurement_df, filename="validation_dataset.tsv")

petab_problem.measurement_df = synthetic_noisy_df
petab.visualize.plot_problem(
    petab_problem=petab_problem,
)
plt.savefig("training_dataset.png")
petab.write_measurement_df(df=petab_problem.measurement_df, filename="training_dataset.tsv")
