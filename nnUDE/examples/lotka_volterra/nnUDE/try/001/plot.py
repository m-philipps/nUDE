from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import copy
from pathlib import Path
from models import pypesto_problems, petab_problems, STAGES, model_names # TRAINING#, pypesto_problems_predict, petab_problems_predict

import matplotlib.pyplot as plt
import pypesto.visualize.model_fit
import pandas as pd
import numpy as np

import pypesto.store
import pypesto.petab
import petab


NEGATIVE_THRESHOLD = -1e-4


def load_results(pypesto_problems):
    results_path = Path(__file__).resolve().parent / "output" / "result"

    pypesto_results = {}
    for model_id, pypesto_problem in pypesto_problems.items():
        pypesto_result = pypesto.store.read_result(results_path / f"{model_id}.hdf5", problem=False, optimize=True)
        pypesto_result.problem = pypesto_problem
        pypesto_results[model_id] = pypesto_result

    return pypesto_results

def plot_waterfall_parameters(model_id, pypesto_result):
    fig, ax = plt.subplots(2, 1)
    pypesto.visualize.waterfall(pypesto_result, ax=ax[0])
    #pypesto.visualize.parameters(pypesto_result, ax=ax[1])
    fig.set_size_inches((10,10))

    fig.suptitle(model_id)


def plot_fit(model_id, pypesto_result, petab_problem):
    simulation_df = pypesto.visualize.model_fit.visualize_optimized_model_fit(
        petab_problem=petab_problem,
        result=pypesto_result,
        pypesto_problem=pypesto_result.problem,
        #amici_solver=pypesto_result.problem.objective.amici_solver,
        return_dict=True
    )['simulation_df']
    plt.close()
    simulation_df['measurement'] = simulation_df['simulation']
    
    
    # In[17]:
    
    
    dfs = {
        'Training data': petab_problem.measurement_df[petab_problem.measurement_df.time <= 20],
        'Validation data': petab_problem.measurement_df[petab_problem.measurement_df.time > 20],
        'Fit': simulation_df,
    }
    plot_types = ['scatter', 'scatter', 'plot']
    plot_settings_list = [
        {
            's': 10**2,
            'marker': '.',
            'alpha': 0.5,
            #'facecolor': 'none',
        },
        {
            's': 10**2,
            'marker': '.',
            #'linewidths': 0.2,
            'alpha': 0.5,
            'facecolor': 'none',
        },
        {},
    ]
    populations = {
        'Prey': {
            'observable_id': 'observable_prey',
            'plot': {
                'color': 'black',
            },
        },
        'Predator': {
            'observable_id': 'observable_predator',
            'plot': {
                'color': 'red',
            },
        },
    }
    
    
    # In[18]:
    
    
    fig, ax = plt.subplots(figsize=(8,3))
    
    prey_lines = []
    predator_lines = []
    for (data_type, df), plot_type, plot_settings in zip(dfs.items(), plot_types, plot_settings_list):
        for population, population_settings in populations.items():
            index = df.observableId == population_settings['observable_id']
            lines = getattr(ax, plot_type)(
                df.time.loc[index],
                df.measurement.loc[index],
                **plot_settings,
                **population_settings['plot'],
                label=population + ": " + data_type,
            )
            if isinstance(lines, list):
                lines = lines[0]
            if population == 'Prey':
                prey_lines.append(lines)
            else:
                predator_lines.append(lines)
    data_legend = ax.legend(prey_lines, list(dfs.keys()), loc='upper right')
    for line_ in data_legend.get_lines():
        line_.set_color('tab:gray')
    ax2 = ax.twinx()
    species_legend = ax2.legend([prey_lines[-1], predator_lines[-1]], list(populations.keys()), loc='upper center')
    if simulation_df.time.max() > 20:
        ax.axvline(20, 0, 1, alpha=0.5, color='k', linestyle='--')

    ax2.tick_params(axis='y', which='both', right=False, left=False, labelright=False, labelleft=False)

    ax.set_xlabel("Time [months]")
    ax.set_ylabel("Population [# / km$^2$]")

    #ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.tick_params(axis='both', which='minor', labelsize=12)


    #fig.set_size_inches((15,5))
    model_name = model_names[model_id].replace('\n', '; ')
    if model_id == "ude":
        model_name = "(A) " + model_name
        #ax.set_xlabel("")
        #ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    elif model_id == "nude":
        model_name = "(B) " + model_name
        ax.set_xlabel("")
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    elif model_id == "integral_regularized_nude":
        model_name = "(C) " + model_name

    #ax.text(1, 0, model_name, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, backgroundcolor='lightgray', bbox={'boxstyle': 'Round, pad=0.01'}, color='lightgray')
    # TODO
    #ax.text(0.99, 0.03, model_name, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, backgroundcolor='lightgray')

    #fig.suptitle(model_name)


def evaluate(x__pypesto_result):
    x, pypesto_result = x__pypesto_result
    if x is None:
        return None
    try:
        evaluation = pypesto_result.problem.objective(
            x[pypesto_result.problem.x_free_indices],
            #sensi_orders=(0,1),
            return_dict=True,
        )
        return evaluation
    except:
        return np.inf
    print("WEIRD ERROR")


def reevaluate(pypesto_result, n_starts: int = None):
    evaluations = []
    if n_starts is not None:
        inputs = [copy.deepcopy((s.x, pypesto_result)) for s in pypesto_result.optimize_result.list[:n_starts]]
    else:
        inputs = [copy.deepcopy((s.x, pypesto_result)) for s in pypesto_result.optimize_result.list]

    with ThreadPoolExecutor(max_workers=48) as pool:
        results = list(
            tqdm(
                pool.map(evaluate, inputs),
                total=len(inputs),
            )
        )

    for start, result in zip(pypesto_result.optimize_result.list, results):
        if result is None:
            continue
        if isinstance(result, float) and np.isinf(result):
            start.fval = np.inf
            continue
        start.fval = result['fval']
        evaluations.append(result)

    pypesto_result.optimize_result.sort()
    return evaluations


def count_signed(evaluations):
    nonnegatives = 0
    negatives = 0
    true_negatives = 0
    min_x = 0
    #infinites = 0
    for e in evaluations:
        x = e['rdatas'][0].x

        if (x >= 0).all():
            nonnegatives += 1
            continue

        if (x < NEGATIVE_THRESHOLD).any():
            true_negatives += 1

        if (x < 0).any():
            negatives += 1
            min_x = min(min_x, x.min())
            continue

        #if 'fval' in e and isinstance(e['fval'], float) and np.isinf(e['fval']):
        #    infinites += 1
    return nonnegatives, negatives, min_x, true_negatives  #, infinites

#def plot_predict(model_id, pypesto_result, petab_problem):
#    pypesto_result.problem = pypesto_problems_predict[model_id]
#    petab_problem = petab_problems_predict[model_id]
#    plot_fit(model_id, pypesto_result, petab_problem)

#STAGE = TRAINING
truncate = 1000

for STAGE in STAGES:
    pypesto_results = load_results(pypesto_problems[STAGE])
    plot_path = Path(__file__).resolve().parent / "output" / "plot" / STAGE
    plot_path.mkdir(exist_ok=True, parents=True)

    if truncate is not None:
        for _, pypesto_result in pypesto_results.items():
            pypesto_result.optimize_result.list = pypesto_result.optimize_result.list[:truncate]

    if True:
        with open(plot_path / "negatives.tsv", "a+") as f:
            f.write(f"model_id\tnonnegatives\tnegatives\tinfinite_fvals\ttotal\tmin_x\ttrue_negatives{NEGATIVE_THRESHOLD}\n")
            #f.write("model_id\tnonnegatives\tnegatives\tinfinite_fvals\tinfinite_fvals_after_reevaluation\ttotal\n")
        
        for model_id, pypesto_result in pypesto_results.items():
            petab_problem = petab_problems[STAGE][model_id]

            infinites = np.isinf(pypesto_result.optimize_result.fval).sum()
            evaluations = reevaluate(pypesto_result=pypesto_result)
            #nonnegatives, negatives, infinites_after_reevaluation = count_signed(evaluations)
            nonnegatives, negatives, min_x, true_negatives = count_signed(evaluations)
            with open(plot_path / "negatives.tsv", "a+") as f:
                f.write(f"{model_id}\t{nonnegatives}\t{negatives}\t{infinites}\t{nonnegatives+negatives+infinites}\t{min_x}\t{true_negatives}\n")
                #f.write(f"{model_id}\t{nonnegatives}\t{negatives}\t{infinites}\t{infinites_after_reevaluation}\t{nonnegatives+negatives+infinites-infinites_after_reevaluation}\n")
        
            waterfall_parameters_path = plot_path / "waterfall_parameters"
            waterfall_parameters_path.mkdir(exist_ok=True, parents=True)
            try:
                plot_waterfall_parameters(model_id=model_id, pypesto_result=pypesto_result)
                plt.savefig(waterfall_parameters_path / f"{model_id}.svg")
            except:
                raise
                #pass
        
            trajectory_plot_path = plot_path / "trajectory"
            trajectory_plot_path.mkdir(exist_ok=True, parents=True)
            plot_fit(model_id=model_id, pypesto_result=pypesto_result, petab_problem=petab_problem)
            #plt.tight_layout()
            plt.savefig(trajectory_plot_path / f"{model_id}.svg")
        
            #fit_path = plot_path / "fit"
            #fit_path.mkdir(exist_ok=True, parents=True)
            #plot_fit(model_id=model_id, pypesto_result=pypesto_result, petab_problem=petab_problem)
            #plt.savefig(fit_path / f"{model_id}.svg")
        
            #predict_path = plot_path / "predict"
            #predict_path.mkdir(exist_ok=True, parents=True)
            #plot_predict(model_id=model_id, pypesto_result=pypesto_result, petab_problem=petab_problem)
            #plt.savefig(predict_path / f"{model_id}.svg")
    if True:
        df = pd.read_csv(plot_path / "negatives.tsv", sep="\t")
        df = df.set_index("model_id")
        
        model_ids = list(df.index)
        
        weight_counts = {
            "Non-negative": [row['nonnegatives'] + row['negatives'] - row['true_negatives-0.0001'] for model_id, row in df.iterrows()],
            "Negative": [row['true_negatives-0.0001'] for model_id, row in df.iterrows()],
            #"Non-evaluable": [row['infinite_fvals'] for model_id, row in df.iterrows()],
        }
        
        fig, ax = plt.subplots(figsize=(6,3))
        bottom = np.zeros(len(df.index))
        
        colors = {
            category: color
            for category, color in zip(weight_counts, ["darkgrey", "orangered", "tab:gray"])
        }
        
        for category, weight_count in weight_counts.items():
            p = ax.bar(
                [model_names[model_id] for model_id in model_ids],
                weight_count,
                0.7,  # width
                label=category,
                bottom=bottom,
                color=colors[category],
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            bottom += weight_count
        
        ax.legend(loc="lower right", reverse=True)

        #ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.set_ylabel("Optimised models [#]")
        
        plt.tight_layout()
        plt.savefig(plot_path / "negatives.svg")
    
    if True:
        #plt.rcParams["mathtext.fontset"] = "dejavuserif"
        if truncate is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
        else:
            fig, ax = plt.subplots(figsize=(20, 5))
        pypesto.visualize.waterfall(
            list(pypesto_results.values()),
            legends=[model_names[model_id].replace('\n', '; ') for model_id in pypesto_results.keys()],
            #order_by_id=True,
            #n_starts_to_zoom=20,
            #offset_y=0,
            ax=ax,
        )
        lines = {}
        line_label0 = None
        for line in ax.get_lines():
            line_label = line.get_label()
            if "UDE" in line_label:
                line_label0 = line_label
                lines[line_label0] = []
            if line_label0 is None or (
                len(lines[line_label0])
                == (truncate if truncate is not None else np.inf)
            ):
                line.set_linewidth(0.5)
                continue
            lines[line_label0].append(line)

        for line_label, label_lines in lines.items():
            for line in label_lines:
                line.set_color("black")
                line.set_linewidth(0.3)
                line.set_marker(".")

                if line_label == "UDE":
                    line.set_color("black")
                    line.set_markeredgecolor("orangered")
                    line.set_markerfacecolor("orangered")
                    line.set_markersize(8)
                elif "tanh" not in line_label and "lambda" not in line_label:
                    line.set_markersize(8)

                if "tanh" in line_label and "lambda" not in line_label:
                    line.set_marker("+")

                if "tanh" not in line_label and "lambda" in line_label:
                    line.set_marker("o")
                    line.set_markersize(4.5)
                    line.set_markeredgewidth(1)
                    line.set_markerfacecolor('none')

                if "tanh" in line_label and "lambda" in line_label:
                    line.set_marker("$\oplus$")
                    line.set_markeredgewidth(0.3)

                if "tanh(10" in line_label:
                    line.set_markeredgecolor("darkgrey")
                    line.set_markerfacecolor("darkgrey")

        ax.legend()
        ax.set_ylabel("$J - J_{min}$ ($J_{min}$ = " + ax.get_ylabel().split("=")[1][1:-1] + ")")
        ax.set_xlabel("Optimised model (ordered by J, ascending)")
        ax.set_title("")

        #ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.tight_layout()
        plt.savefig(plot_path / "waterfall_comparison.svg")
