from bengalese_finch.models.probzip import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import os

#################################################################################

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_synthetic_balancetest/'
figs_dir = 'bengalese_finch/analyses/probzip/figs/synthetic/'
groundtruth_models_dir = 'bengalese_finch/analyses/probzip/groundtruth_models_synthetic/'

n_runs = 8
def get_models_results(models_dir):
    # Construct the models and results dictionaries
    models = {}
    models_results = {}
    for run_i in range(n_runs):
        models[run_i] = []
        models_results[run_i] = []
        for model_i in range(10):
            try:
                with open(f'{models_dir}model_balancetest_{run_i}_{model_i}.pkl', 'rb') as f:
                    compressor = pickle.load(f)
            except FileNotFoundError:
                print(f'Model file not found for groundtruth model {run_i}, model {model_i}. Skipping.')
                continue
            try:
                with open(f'{models_dir}results_balancetest_{run_i}_{model_i}.pkl', 'rb') as f:
                    results_dict = pickle.load(f)
            except FileNotFoundError:
                print(f'Results file not found for groundtruth model {run_i}, model {model_i}. Skipping.')
                continue

            models[run_i].append(compressor)
            models_results[run_i].append(results_dict)

    return models, models_results

# TODO(noemielteto): Goes in analysis library
def get_best_model_i(results):
    best_model_i = np.argmin([r['mdl'][-1] for r in results])
    return best_model_i

models, models_results = get_models_results(models_dir)

best_model_i = [get_best_model_i(models_results[run_i]) for run_i in range(n_runs)]
best_models = [models[run_i][best_model_i[run_i]] for run_i in range(n_runs)]

ab_proportions = [sum([1 for x in compressor.generate_dataset(size=1000) if x =='<ab>']) for compressor in best_models]
