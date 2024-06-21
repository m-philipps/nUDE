from pathlib import Path

import pypesto
import pypesto.petab
import pypesto.optimize
import pypesto.visualize.model_fit

import pandas as pd
import matplotlib.pyplot as plt

import sys

boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))

from helpers import *
from hyperparameters import *
from models import pypesto_importers, pypesto_problems

# load result
output_path = Path(__file__).resolve().parent / "output" / GRID_ID
result_path = Path(__file__).resolve().parent / "result" / GRID_ID
result_path.mkdir(exist_ok=True, parents=True)
(result_path / "fits").mkdir(exist_ok=True)

ude_result = pypesto.store.read_result(
    output_path / "ude_result_likelihood.hdf5",
    optimize=True,
    profile=False,
    sample=False,
)
nude_result = pypesto.store.read_result(
    output_path / "nude_result_likelihood.hdf5",
    optimize=True,
    profile=False,
    sample=False,
)
bnude_result = pypesto.store.read_result(
    output_path / "bnude_result_likelihood.hdf5",
    optimize=True,
    profile=False,
    sample=False,
)

# additional metrics
fp_losses = result_path / "losses.csv"
calculate_additional_metrics(
    results=(ude_result, nude_result, bnude_result),
    importers=(
        pypesto_importers[UDE][0],
        pypesto_importers[NUDE][0],
        pypesto_importers[BNUDE][0],
    ),
    fp=fp_losses,
)
df_losses = pd.read_csv(fp_losses)
barplot_non_negative(
    df=df_losses,
    fp=result_path / "non_negatives.png",
    n_starts=N_STARTS,
)
show_computation(
    df=df_losses,
    labels=LABELS,
    fp=result_path / "computation.png",
)
paper_figure_nonneg_runningtime(
    df_losses,
    LABELS,
    fp=result_path / "tradeoff.png",
)

# compare waterfall plots
pypesto.visualize.waterfall(
    [ude_result, nude_result, bnude_result],
    legends=list(LABELS),
    n_starts_to_zoom=20,
)
plt.savefig(result_path / "waterfalls.png")

# compare parameters
for label, result in zip(LABELS, (ude_result, nude_result, bnude_result)):
    parameters_plot(
        result,
        label,
        n_starts_to_show=100,
        fp=result_path / f"parameters_{label}.png",
    )

# best fit
for label, result in zip(LABELS, (ude_result, nude_result, bnude_result)):
    pypesto_importer = pypesto_importers[label][0]
    pypesto_problem = pypesto_problems[label][0]

    fp = result_path / f"best_fit_{label}.png"
    plot_observables_fit_boehm(
        label=label,
        petab_problem=pypesto_importer.petab_problem,
        result=result,
        pypesto_problem=pypesto_problem,
        pypesto_result_id=0,
        fp=fp,
    )
    # fit with noise
    fp = result_path / f"best_fit_{label}_noise.png"
    plot_observables_fit_boehm(
        label=label,
        petab_problem=pypesto_importer.petab_problem,
        result=result,
        pypesto_problem=pypesto_problem,
        pypesto_result_id=0,
        show_noise=True,
        fp=fp,
    )

    # 10 best fits
    for i in range(10):
        fp = result_path / "fits" / f"best_fit_{label}_{i}.png"
        plot_observables_fit_boehm(
            label=label,
            petab_problem=pypesto_importer.petab_problem,
            result=result,
            pypesto_problem=pypesto_problem,
            pypesto_result_id=i,
            show_noise=True,
            fp=fp,
        )

        # simulate all state variables
        fp = result_path / "fits" / f"simulation_{label}_{i}.png"
        plot_state_trajectories_boehm(
            label=label,
            result=result,
            pypesto_problem=pypesto_problem,
            pypesto_result_id=i,
            fp=fp,
        )

    fp = result_path / f"best_state_trajectories_{label}.png"
    plot_state_trajectories_boehm(
        label=label,
        result=result,
        pypesto_problem=pypesto_problem,
        pypesto_result_id=0,
        fp=fp,
    )

    # residuals probability plots
    fp = result_path / f"residuals_{label}.png"
    plot_residual(
        label=label,
        result=result,
        pypesto_importer=pypesto_importer,
        pypesto_problem=pypesto_problem,
        pypesto_result_id=0,
        fp=fp,
    )

# plot ensemble best fits
for n in [20, 50]:
    fp = result_path / f"ensembles_{n}.png"
    plot_ensembles(
        results=[ude_result, nude_result, bnude_result],
        pypesto_importers=[
            pypesto_importers[UDE][0],
            pypesto_importers[NUDE][0],
            pypesto_importers[BNUDE][0],
        ],
        pypesto_problems={l: pypesto_problems[l][0] for l in LABELS},
        labels=LABELS,
        n=n,
        fp=fp,
    )

# Compare simulations
compare_state_trajectories_boehm(
    labels=LABELS,
    results=(ude_result, nude_result, bnude_result),
    pypesto_problems=(
        pypesto_problems[UDE][0],
        pypesto_problems[NUDE][0],
        pypesto_problems[BNUDE][0],
    ),
    pypesto_result_id=0,
    fp=result_path / "best_state_trajectories.png",
)
