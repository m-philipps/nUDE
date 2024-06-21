from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import pypesto
import pypesto.petab
import pypesto.optimize
import pypesto.visualize.model_fit

import sys
boehm_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(boehm_dir))

from helpers import *
from hyperparameters import *

reg_names = [str(r) for r in REGULARIZATION_STRENGTHS]

# load results

output_path = Path(__file__).resolve().parent / "output"
result_path = Path(__file__).resolve().parent / "result"
result_reg_path = result_path / "regularization"
result_reg_path.mkdir(exist_ok=True)

losses = {
    rs: pd.read_csv(result_path / f"reg_{str(rs)}" / "losses.csv")
    for rs in REGULARIZATION_STRENGTHS
}

ude_results = {
    r: pypesto.store.read_result(
        output_path / f"reg_{r}" / f"ude_result_likelihood.hdf5",
        optimize=True,
        profile=False,
        sample=False,
    )
    for r in REGULARIZATION_STRENGTHS
}
nude_results = {
    r: pypesto.store.read_result(
        output_path / f"reg_{r}" / f"nude_result_likelihood.hdf5",
        optimize=True,
        profile=False,
        sample=False,
    )
    for r in REGULARIZATION_STRENGTHS
}
bnude_results = {
    r: pypesto.store.read_result(
        output_path / f"reg_{r}" / f"bnude_result_likelihood.hdf5",
        optimize=True,
        profile=False,
        sample=False,
    )
    for r in REGULARIZATION_STRENGTHS
}


ude_results_nmse = deepcopy(ude_results)
for reg, result in ude_results_nmse.items():
    for res in result.optimize_result:
        nmse = (
            losses[reg]
            .query("method == 'ude'")
            .query(f"start_id == {res['id']}")["nmse"]
            .item()
        )
        res["fval"] = nmse
    result.optimize_result.sort()
nude_results_nmse = deepcopy(nude_results)
for reg, result in nude_results_nmse.items():
    for res in result.optimize_result:
        nmse = (
            losses[reg]
            .query("method == 'nude'")
            .query(f"start_id == {res['id']}")["nmse"]
            .item()
        )
        res["fval"] = nmse
    result.optimize_result.sort()
bnude_results_nmse = deepcopy(bnude_results)
for reg, result in bnude_results_nmse.items():
    for res in result.optimize_result:
        nmse = (
            losses[reg]
            .query("method == 'nude'")
            .query(f"start_id == {res['id']}")["nmse"]
            .item()
        )
        res["fval"] = nmse
    result.optimize_result.sort()

# waterfall plots

pypesto.visualize.waterfall(
    list(ude_results.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
)
plt.savefig(result_path / "regularization" / f"waterfalls_ude.png")
pypesto.visualize.waterfall(
    list(nude_results.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
)
plt.savefig(result_path / "regularization" / f"waterfalls_nude.png")
pypesto.visualize.waterfall(
    list(bnude_results.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
)
plt.savefig(result_path / "regularization" / f"waterfalls_bnude.png")


# waterfall by best NMSE
pypesto.visualize.waterfall(
    list(ude_results_nmse.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
    y_limits=(0.1, 1e12),
)
plt.savefig(result_path / "regularization" / f"waterfalls_nmse_ude.png")
pypesto.visualize.waterfall(
    list(nude_results_nmse.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
    y_limits=(0.1, 1e12),
)
plt.savefig(result_path / "regularization" / f"waterfalls_nmse_nude.png")
pypesto.visualize.waterfall(
    list(bnude_results_nmse.values()),
    legends=reg_names,
    n_starts_to_zoom=20,
    y_limits=(0.1, 1e12),
)
plt.savefig(result_path / "regularization" / f"waterfalls_nmse_bnude.png")


# best loss by regularizarion

fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
metrics = ["fval", "nmse"]
for metric, axis in zip(metrics, ax):
    df = pd.DataFrame(
        {
            rs: losses[rs].groupby(by="method").min()[metric]
            for rs in REGULARIZATION_STRENGTHS
        }
    )
    df.columns = reg_names
    df.T.plot(
        marker=".",
        linestyle="none",
        markersize=12,
        ax=axis,
        legend=(metric == metrics[0]),
    )
    axis.set_title(f"Best {metric}")
    axis.set_xlabel("Regularization strength")
fig.tight_layout()
fig.savefig(result_path / "regularization" / "best_losses.png")

