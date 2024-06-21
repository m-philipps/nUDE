from pathlib import Path
from typing import Tuple, List
from copy import deepcopy

import petab
from petab.C import *

# import benchmark_models_petab
import amici
from amici.petab_objective import rdatas_to_simulation_df
import pypesto
import pypesto.petab

# import nnUDE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scipy.stats as stats


UDE = "ude"
NUDE = "nude"
BNUDE = "bnude"
LABELS = (UDE, NUDE, BNUDE)

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

observable_ids = ["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"]

observable_colors = ["tomato", "limegreen", "mediumorchid"]


def get_muted_petab_problem() -> petab.Problem:
    # get the modified boehm problem without nucpapa export
    fp_yaml = (
        Path(__file__).parent
        / "petab_muted_papa_export"
        / "Boehm_JProteomeRes2014.yaml"
    )
    petab_problem = petab.Problem.from_yaml(fp_yaml)
    return petab_problem


def fix_objective(objective, strict: bool = False) -> None:
    objective.amici_solver.setMaxSteps(int(1e5))
    objective.amici_solver.setAbsoluteTolerance(1e-14)
    objective.amici_solver.setRelativeTolerance(1e-12)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
    objective.amici_solver.setMaxTime(30)
    if strict:
        objective.amici_solver.setAbsoluteTolerance(1e-15)
        objective.amici_solver.setRelativeTolerance(1e-13)


AMICI_EXITFLAG_MAPPING = {
    0: "Optimizer did not run",
    -1: "Reached maximum number of allowed iterations",
    -2: "Expected to reach maximum allowed time in next iteration",
    -3: "Encountered non-finite fval/grad/hess",
    -4: "Exceeded specified boundaries",
    -5: "Trust Region Radius too small to proceed",
    1: "Converged according to fval difference",
    2: "Converged according to x difference",
    3: "Converged according to gradient norm",
}


def regularize_petab_problem_with_l2(petab_problem, regularization_strength):
    "L2 regularization for parameters whose IDs start with 'neural_network'."
    df = petab_problem.parameter_df
    df[OBJECTIVE_PRIOR_TYPE] = deepcopy(df[INITIALIZATION_PRIOR_TYPE])
    df[OBJECTIVE_PRIOR_PARAMETERS] = deepcopy(df[INITIALIZATION_PRIOR_PARAMETERS])
    df.loc[
        [s.startswith("neural_network") for s in df.index], OBJECTIVE_PRIOR_PARAMETERS
    ] = f"0;{1/regularization_strength}"
    petab_problem.parameter_df = df


def update_fvals(fp_input, fp_output, pypesto_problem):
    """Create a new result file from the given result at fp_input with fvals for the given problem."""
    result = pypesto.store.read_result(
        fp_input, optimize=True, profile=False, sample=False
    )
    for res in result.optimize_result:
        if not isinstance(res["x"], np.ndarray):
            # if no x available, skip
            res["fval"] = np.nan
            continue
        # get reduced parameter vector without fixed parameters
        x = pypesto_problem.get_reduced_vector(res["x"])
        res["fval"] = pypesto_problem.objective(x)
    result.optimize_result.sort()
    pypesto.store.write_result(
        result=result, filename=fp_output, optimize=True, overwrite=True
    )


# == evaluation ==


def calculate_additional_metrics(
    results: Tuple[pypesto.Result], importers: Tuple[pypesto.petab.PetabImporter], fp
):
    numeric_correction_term = 1e-13  # one magnitude more than abs. tolerance
    ref_means = REFERENCE_SOLUTION[:, 1:].mean(axis=0)
    dfs = []
    for label, result, importer in zip(LABELS, results, importers):
        if not result:
            continue
        pypesto_objective = importer.create_objective()
        fix_objective(pypesto_objective, strict=True)
        timepoints = np.arange(241.0)
        objective = pypesto_objective.set_custom_timepoints(
            timepoints_global=timepoints
        )
        for res in result.optimize_result.list:
            parameters = res["x"]
            rdatas = objective(parameters, return_dict=True)["rdatas"][0]
            res["non_negative"] = not (rdatas.x < -numeric_correction_term).any()
            # calculate mse for hidden states
            mse_by_state = (np.square(REFERENCE_SOLUTION[:, 1:] - rdatas.x)).mean(
                axis=0
            )
            for e, s in zip(mse_by_state, species_ids):
                res["mse_" + s] = e
            # total mse
            mse = mse_by_state.mean()
            nmse = (mse_by_state / ref_means).mean()
            if np.isnan(mse):
                mse = np.inf
            if np.isnan(nmse):
                nmse = np.inf
            res["mse"] = mse
            res["nmse"] = nmse
        # save different errors for exploration
        df = pd.DataFrame(
            {
                **{
                    "start_id": [res["id"] for res in result.optimize_result.list],
                    "fval": [res["fval"] for res in result.optimize_result.list],
                    "mse": [res["mse"] for res in result.optimize_result.list],
                    "nmse": [res["nmse"] for res in result.optimize_result.list],
                    "time": [res["time"] for res in result.optimize_result.list],
                    "n_iterations": [
                        res["n_fval"] for res in result.optimize_result.list
                    ],
                    "non_negative": [
                        res["non_negative"] for res in result.optimize_result.list
                    ],
                    "exitflag": [
                        res["exitflag"] for res in result.optimize_result.list
                    ],
                },
                **{
                    f"mse_{s}": [res[f"mse_{s}"] for res in result.optimize_result.list]
                    for s in species_ids
                },
            }
        )
        df["method"] = label
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(fp, index=False)


def barplot_non_negative(df: pd.DataFrame, fp, n_starts: int = 1000):
    """
    Creates a bar plot for the fraction of UDE/NUDE starts that stayed non-negative.

    nn_ude, nn_nude expects a tuple (# stayed non-negative, # total)
    """
    nn_ude = df[df.method == UDE]["non_negative"].sum()
    nn_nude = df[df.method == NUDE]["non_negative"].sum()
    nn_bnude = df[df.method == BNUDE]["non_negative"].sum()
    plt.bar(x=LABELS, height=[nn_ude, nn_nude, nn_bnude], color="darkgrey")
    plt.hlines(n_starts, -0.5, 2.5, "k")
    plt.title(f"Stayed non-negative: {nn_ude} vs. {nn_nude}, {nn_bnude}")
    plt.tight_layout()
    plt.xticks(rotation=0, fontsize=12)
    plt.savefig(fp)


# == visualization ==

# reference
REFERENCE_OPTIMUM = [
    {
        "legend": None,
        "x": [
            -1.56891759,
            -4.99970489,
            -2.20969878,
            -1.78600655,
            4.99011401,
            4.19773549,
            0.693,
            0.58575527,
            0.81898282,
            0.4986844,
            0.107,
        ],
        "fval": 138.22199760826,
        "color": [0.0, 0.5, 0.0, 0.9],
        "auto_color": True,
    }
]
REFERENCE_SOLUTION = np.loadtxt(
    Path(__file__).resolve().parent / "boehm_reference_solution.dat"
)


def parameters_plot(
    result: pypesto.Result,
    label: str,
    n_starts_to_show: int = 100,
    fp=None,
) -> None:
    ids_mechanistic_parameters = [0, 1, 2, 3, 4, 5, 7, 8, 9]  # 6 is fixed
    pypesto.visualize.parameters(
        result,
        parameter_indices=ids_mechanistic_parameters,
        reference=REFERENCE_OPTIMUM,
        start_indices=n_starts_to_show,
    )
    plt.title(f"Estimated parameters {label.upper()}: Best {n_starts_to_show}")
    if fp:
        plt.savefig(fp)
    plt.show()


def plot_observables_fit_boehm(
    label: str,
    petab_problem: petab.Problem,
    result: pypesto.Result,
    pypesto_problem: pypesto.Problem,
    pypesto_result_id: int = 0,
    show_noise: bool = False,
    fp=None,
):
    # simulate with more time points
    objective = pypesto_problem.objective
    timepoints = np.arange(241.0)
    objective_ = objective.set_custom_timepoints(timepoints_global=timepoints)
    parameters = pypesto_problem.get_reduced_vector(
        result.optimize_result[pypesto_result_id]["x"]
    )
    rdatas = objective_(parameters, return_dict=True)["rdatas"][0]
    param_dict = dict(
        zip(pypesto_problem.x_names, result.optimize_result[pypesto_result_id]["x"])
    )

    fig, ax = plt.subplots()
    for i, (obs, color) in enumerate(zip(observable_ids, observable_colors)):
        # measurements
        df = petab_problem.measurement_df.query("observableId == @obs")
        ax.scatter(
            df.time, df.measurement, s=4, color=color, label=f"{obs}: measurement"
        )
        # simulation
        y = rdatas.y[:, i]
        ax.plot(rdatas.t, y, color=color, label=f"{obs}: simulation")

        # show noise
        if show_noise:
            sd_nominal = 10 ** param_dict["sd_" + obs]
            ax.fill_between(
                rdatas.t,
                (y - sd_nominal),
                (y + sd_nominal),
                color=color,
                alpha=0.2,
                edgecolor=None,
                label=f"$\sigma$ = {round(sd_nominal, 2)}",
            )

    ax.legend(loc="upper right")
    fig.set_size_inches((6, 4))
    fig.suptitle(f"Best Fit from {label.upper()}")
    fig.tight_layout()
    if fp:
        plt.savefig(fp)
    plt.show()


def plot_residual(
    label: str,
    result,
    pypesto_importer: pypesto.petab.importer,
    pypesto_problem: pypesto.Problem,
    pypesto_result_id: int = 0,
    fp=None,
):
    # measurements
    meas = pypesto_importer.petab_problem.measurement_df
    # simulate
    p = pypesto_problem.get_reduced_vector(
        result.optimize_result[pypesto_result_id]["x"]
    )
    rdatas = pypesto_problem.objective(p, return_dict=True)["rdatas"]
    sim = rdatas_to_simulation_df(
        rdatas,
        model=pypesto_problem.objective.amici_model,
        measurement_df=meas,
    )
    # residuals
    dfres = pd.merge(
        meas[[OBSERVABLE_ID, TIME, MEASUREMENT]],
        sim[[OBSERVABLE_ID, TIME, SIMULATION]],
        on=[OBSERVABLE_ID, TIME],
    )
    dfres["residual"] = dfres[MEASUREMENT] - dfres[SIMULATION]

    # get parameter dict
    param_dict = dict(
        zip(pypesto_problem.x_names, result.optimize_result[pypesto_result_id]["x"])
    )

    # generate the plot
    fig, ax = plt.subplots()
    fig.suptitle("Residuals " + label.upper())
    for obs, c in zip(observable_ids, observable_colors):
        # get estimated noise parameter
        sd_nominal = 10 ** param_dict["sd_" + obs]
        # normal distribution
        dist = stats.norm(loc=0, scale=sd_nominal)
        # quantiles of normal distribution
        n_quantiles = dfres[OBSERVABLE_ID].value_counts()[obs] + 1
        assert n_quantiles == 17
        quantiles_normal = [dist.ppf(q / n_quantiles) for q in range(1, n_quantiles)]
        # calculate residuals
        residuals = sorted(dfres.query("observableId == @obs")["residual"])
        ax.scatter(quantiles_normal, residuals, label=obs, c=c)
        ax.set_xlabel("Normal Distribution")
        ax.set_ylabel("Residuals")
    ax.legend()
    # add diagonal
    maxi = max(sorted(dfres["residual"]) + quantiles_normal)
    mini = min(sorted(dfres["residual"]) + quantiles_normal)
    ax.plot([mini, maxi], [mini, maxi], ls="--", c="darkgrey")
    fig.tight_layout()
    if fp:
        plt.savefig(fp)
    plt.show()


def plot_state_trajectories_boehm(
    label: str,
    result: pypesto.Result,
    pypesto_problem: pypesto.Problem,
    pypesto_result_id: int = 0,
    fp=None,
):
    # simulate with more time points
    objective = pypesto_problem.objective
    timepoints = np.arange(241.0)
    objective_ = objective.set_custom_timepoints(timepoints_global=timepoints)
    parameters = pypesto_problem.get_reduced_vector(
        result.optimize_result[pypesto_result_id]["x"]
    )
    rdatas = objective_(parameters, return_dict=True)["rdatas"][0]

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 6))
    for i, tup in enumerate(zip(ax.flatten(), species_ids)):
        axis, state_name = tup
        axis.plot(rdatas.t, rdatas.x[:, i], label="simulation")
        axis.plot(
            REFERENCE_SOLUTION[:, 0],
            REFERENCE_SOLUTION[:, 1 + i],
            c="dimgrey",
            linestyle="--",
            label="reference",
        )
        axis.set_title(state_name, size="medium")
    ax[0][0].legend()
    fig.suptitle(f"Simulate hidden states with best {label.upper()} model")
    fig.tight_layout()
    if fp:
        plt.savefig(fp)
    plt.show()


def compare_state_trajectories_boehm(
    labels: Tuple[str],
    results: Tuple[pypesto.Result],
    pypesto_problems: Tuple[pypesto.Problem],
    pypesto_result_id: int = 0,
    fp=None,
):
    timepoints = np.arange(241.0)
    rdatas = {}

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 6))
    for label, result, problem in zip(labels, results, pypesto_problems):
        objective = problem.objective
        timepoints = np.arange(241.0)
        objective_ = objective.set_custom_timepoints(timepoints_global=timepoints)
        parameters = problem.get_reduced_vector(
            result.optimize_result[pypesto_result_id]["x"]
        )
        rdatas[label] = objective_(parameters, return_dict=True)["rdatas"][0]

    for i, tup in enumerate(zip(ax.flatten(), species_ids)):
        axis, state_name = tup
        for l in labels:
            axis.plot(rdatas[l].t, rdatas[l].x[:, i], label=l + " simulation")
        axis.plot(
            REFERENCE_SOLUTION[:, 0],
            REFERENCE_SOLUTION[:, 1 + i],
            c="dimgrey",
            linestyle="--",
            label="reference",
        )
        axis.set_title(state_name, size="medium")
    ax[0][0].legend()
    fig.suptitle(f"Simulate hidden states with best models")
    fig.tight_layout()
    if fp:
        plt.savefig(fp)
    plt.show()


def plot_ensembles(
    results: List[pypesto.result.Result],
    pypesto_importers: list,
    pypesto_problems: list,
    labels: List[str],
    n,
    fp=None,
):
    ensembles = [
        pypesto.ensemble.Ensemble.from_optimization_endpoints(
            res,
            max_size=n,
        )
        for res in results
    ]
    predictors = [
        pypesto.predict.AmiciPredictor(
            pypesto_problems[label].objective.set_custom_timepoints([np.arange(240.0)]),
        )
        for label in labels
    ]
    predictions = [
        ens.predict(predictor) for ens, predictor in zip(ensembles, predictors)
    ]
    # plot
    fig, ax = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(12, 3))
    for index, (label, importer) in enumerate(zip(labels, pypesto_importers)):
        for i, (obs, color) in enumerate(zip(observable_ids, observable_colors)):
            df = importer.petab_problem.measurement_df.query("observableId == @obs")
            for res in predictions[index].prediction_results:
                ax[index].plot(
                    np.arange(240.0),
                    res.conditions[0].output[:, i],
                    color=color,
                    label=f"{obs}: simulation",
                    alpha=0.5,
                )
            ax[index].scatter(
                df.time,
                df.measurement,
                s=12,
                color=color,
                label=f"{obs}: measurement",
                edgecolors="k",
                zorder=2.5,
            )
            ax[index].set_title(label[:-3] + "UDE")
    # legend
    ax[2].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", color=c, label=l)
            for c, l in zip(observable_colors, observable_ids)
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="w",
                markeredgecolor="k",
                label="measurement",
            ),
            Line2D([0], [0], color="k", label="simulation"),
        ],
        bbox_to_anchor=(1.05, 0.8),
    )
    fig.tight_layout()
    if fp:
        fig.savefig(fp)


def paper_figure_nonneg_runningtime(df, labels, fp):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    xlabels = [l[:-3] + "UDE" for l in labels]

    # show non-negatives
    nn_ude = df[df.method == UDE]["non_negative"].sum()
    nn_nude = df[df.method == NUDE]["non_negative"].sum()
    nn_bnude = df[df.method == BNUDE]["non_negative"].sum()
    ax[0].bar(x=xlabels, height=[nn_ude, nn_nude, nn_bnude], color="darkgrey")
    ax[0].hlines(1000, -0.5, 2.5, "k")
    ax[0].set_title(f"Stayed non-negative: {nn_ude} vs. {nn_nude}, {nn_bnude}")
    # ax[0].set_xticks(rotation=0, fontsize=12)

    # compare running time
    running_times = [[rt / 60 for rt in df[df.method == l]["time"]] for l in labels]
    ax[1].boxplot(
        running_times,
        labels=xlabels,
    )
    ax[1].set_yscale("log")
    ax[1].set_title("Running time [min]")

    # save
    fig.tight_layout()
    fig.savefig(fp, dpi=400)


def show_computation(df, labels, fp):
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    # get the number of iterations per approach
    # assuming that n_fval is that
    n_iters = [df[df.method == l]["n_iterations"] for l in labels]
    ax[0][0].hist(
        n_iters,
        alpha=0.8,
        label=labels,
    )
    ax[0][0].set_title("Histogram: Number of Iterations")
    plt.legend()
    ax[0][0].set_yscale("log")

    # compare running time
    running_times = [[rt / 60 for rt in df[df.method == l]["time"]] for l in labels]
    axis_labels = []
    for l, rt in zip(labels, running_times):
        axis_labels.append(f"{l} (max {round(max(rt) / 60, 2)} h)")

    ax[0][1].hist(
        running_times,
        alpha=0.8,
        label=axis_labels,
    )
    ax[0][1].set_yscale("log")
    ax[0][1].set_title("Histogram: Running time [min]")
    ax[0][1].legend()

    # calculate average simulation time
    df["avg_simulation_time"] = df["time"] / df["n_iterations"]
    sim_times = [df[df.method == l]["avg_simulation_time"] for l in labels]
    sim_time_means = df.groupby(by="method")["avg_simulation_time"].mean()
    axis_labels = [f"{l} (mean {round(sim_time_means[l], 2)} s)" for l in labels]

    ax[1][0].hist(
        sim_times,
        alpha=0.8,
        label=axis_labels,
    )
    ax[1][0].set_yscale("log")
    ax[1][0].set_title("Histogram: Average simulation time per run")
    ax[1][0].legend()
    ax[1][0].set_xlabel("seconds")

    # compare exit messages
    exit_messages = [
        df[df.method == l]["exitflag"].map(AMICI_EXITFLAG_MAPPING) for l in labels
    ]
    ax[1][1].hist(
        exit_messages,
        alpha=0.8,
        label=labels,
    )
    # locations = [(text._x, text._y) for text in ax[2].get_xticklabels()]
    xticklabels = [text._text for text in ax[1][1].get_xticklabels()]

    # ax[2].set_xticklabels(locations, xticklabels)
    def insert_newlines(string, every=20):
        return "\n".join(string[i : i + every] for i in range(0, len(string), every))

    xticklabels = [insert_newlines(l) for l in xticklabels]
    ax[1][1].set_xticklabels(xticklabels, rotation=15)  # , wrap=True)
    ax[1][1].set_title("Histogram: Exit messages")
    ax[1][1].legend()
    ax[1][1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(fp)


def best_mse_by_species(df: pd.DataFrame, fp, metric="fval"):
    """
    Plot best MSE by state for UDE/NUDE.

    Parameters
    ----------
    df:
        pd.DataFrame with the losses
    fp:
        filepath for storing the resulting plot
    metric:
        by which metric to select the best start, defaults to 'fval'
    """
    markeroptions = {
        UDE: {
            "fval": {
                "marker": "o",
                "markerfacecolor": "cornflowerblue",
                "markeredgecolor": "cornflowerblue",
            },
            "nmse": {
                "marker": "s",
                "markerfacecolor": "cornflowerblue",
                "markeredgecolor": "cornflowerblue",
            },
        },
        NUDE: {
            "fval": {
                "marker": "o",
                "markerfacecolor": "tab:orange",
                "markeredgecolor": "tab:orange",
            },
            "nmse": {
                "marker": "s",
                "markerfacecolor": "tab:orange",
                "markeredgecolor": "tab:orange",
            },
        },
        BNUDE: {
            "fval": {
                "marker": "o",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
            },
            "nmse": {
                "marker": "s",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
            },
        },
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in LABELS:
        score = df.iloc[df.groupby(by="method").idxmin().loc[method, metric]]
        ax.plot(
            species_ids,
            [score["mse_" + s] for s in species_ids],
            **markeroptions[method][metric],
            markersize=12,
            linewidth=0,
            # markersize=4,
            label=f"{method} best {metric}",
            alpha=0.8,
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-6)
    ax.legend()
    ax.set_title("MSE by species")
    fig.savefig(fp)
    plt.close()
