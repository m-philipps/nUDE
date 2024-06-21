import copy
from pathlib import Path
from models import pypesto_problems, petab_problems, pypesto_problems_predict, petab_problems_predict

import matplotlib.pyplot as plt
import pypesto.visualize.model_fit

import pypesto.store
import pypesto.petab
import petab


def load_results(pypesto_problems):
    results_path = Path(__file__).resolve().parent / "output" / "result"

    pypesto_results = {}
    for model_id, pypesto_problem in pypesto_problems.items():
        pypesto_result = pypesto.store.read_result(results_path / f"{model_id}.hdf5")
        pypesto_result.problem = pypesto_problem
        pypesto_results[model_id] = pypesto_result

    return pypesto_results

def plot_waterfall(model_id, pypesto_result):
    fig, ax = plt.subplots()
    pypesto.visualize.waterfall(pypesto_result, ax=ax)
    #pypesto.visualize.parameters(pypesto_result, ax=ax[1])
    #fig.set_size_inches((10,10))

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
        'measurement': petab_problem.measurement_df,
        'simulation': simulation_df,
    }
    plotters = {
        'scatter': {
            's':4,
        },
        'plot': {},
    }
    populations = {
        'prey': {
            'observable_id': 'observable_prey',
            'plot': {
                'color': 'black',
            },
        },
        'predator': {
            'observable_id': 'observable_predator',
            'plot': {
                'color': 'red',
            },
        },
    }
    
    
    # In[18]:
    
    
    fig, ax = plt.subplots()
    
    for (data_type, df), (plot_type, plot_settings) in zip(dfs.items(), plotters.items()):
        for population, population_settings in populations.items():
            index = df.observableId == population_settings['observable_id']
            getattr(ax, plot_type)(
                df.time.loc[index],
                df.measurement.loc[index],
                **plot_settings,
                **population_settings['plot'],
                label=population + ": " + data_type,
            )
    ax.legend()
    fig.set_size_inches((15,5))
    fig.suptitle(model_id)


def plot_predict(model_id, pypesto_result):
    pypesto_result.problem = pypesto_problems_predict[model_id]
    petab_problem = petab_problems_predict[model_id]
    plot_fit(model_id, pypesto_result, petab_problem)


pypesto_results = load_results(pypesto_problems)

simulations = {}
stayed_non_negative = {}

for model_id, pypesto_result in pypesto_results.items():
    petab_problem = petab_problems_predict[model_id]
    pypesto_problem = pypesto_problems_predict[model_id]

    waterfall_parameters_path = plot_path / "waterfall_parameters"
    waterfall_parameters_path.mkdir(exist_ok=True, parents=True)
    plot_waterfall_parameters(model_id=model_id, pypesto_result=pypesto_result)
    plt.savefig(waterfall_parameters_path / f"{model_id}.svg")

    fit_path = plot_path / "fit"
    fit_path.mkdir(exist_ok=True, parents=True)
    plot_fit(model_id=model_id, pypesto_result=pypesto_result, petab_problem=petab_problem)
    plt.savefig(fit_path / f"{model_id}.svg")

    simulations[model_id] = []
    stayed_non_negative[model_id] = []
    for start in pypesto_result.optimize_result.list:
        res = pypesto_problems_predict[model_id].objective(start.x[petab_problem.x_free_indices], return_dict=True)
        start.fval = res["fval"]

        simulations[model_id].append(res)
        stayed_non_negative[model_id].append(
            all([
                (cond.x >= 0).all()
                for cond in res["rdatas"]
            ])
        )
    print(f"{model_id} stayed non negative: {sum(stayed_non_negative[model_id])}/{len(stayed_non_negative[model_id])}")

    predict_path = plot_path / "predict"
    predict_path.mkdir(exist_ok=True, parents=True)
    plot_predict(model_id=model_id, pypesto_result=pypesto_result, petab_problem=petab_problem)
    plt.savefig(predict_path / f"{model_id}.svg")
