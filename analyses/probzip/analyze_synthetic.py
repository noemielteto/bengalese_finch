from bengalese_finch.models.probzip import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import os

#################################################################################

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_synthetic/'
figs_dir = 'bengalese_finch/analyses/probzip/figs/synthetic/'
groundtruth_models_dir = 'bengalese_finch/analyses/probzip/groundtruth_models_synthetic/'

n_models = len([name for name in os.listdir(groundtruth_models_dir) if os.path.isfile(os.path.join(groundtruth_models_dir, name))])
dataset_sizes = [10, 100, 1000]
tasks = [(i, size) for i in range(n_models) for size in dataset_sizes]

def get_models_results(models_dir):
    # Construct the models and results dictionaries
    models = {}
    models_results = {}
    for gt_model_i in range(n_models):
        models[gt_model_i] = {}
        models_results[gt_model_i] = {}
        for dataset_size in dataset_sizes:
            models[gt_model_i][dataset_size] = []
            models_results[gt_model_i][dataset_size] = []
            for model_i in range(10):
                try:
                    with open(f'{models_dir}model_{gt_model_i}_{dataset_size}_{model_i}.pkl', 'rb') as f:
                        compressor = pickle.load(f)
                except FileNotFoundError:
                    print(f'Model file not found for groundtruth model {gt_model_i}, dataset size {dataset_size}, model {model_i}. Skipping.')
                    continue
                try:
                    with open(f'{models_dir}results_{gt_model_i}_{dataset_size}_{model_i}.pkl', 'rb') as f:
                        results_dict = pickle.load(f)
                except FileNotFoundError:
                    print(f'Results file not found for groundtruth model {gt_model_i}, dataset size {dataset_size}, model {model_i}. Skipping.')
                    continue

                models[gt_model_i][dataset_size].append(compressor)
                models_results[gt_model_i][dataset_size].append(results_dict)

    return models, models_results

# TODO(noemielteto): Goes in analysis library
def get_best_model_i(results):
    best_model_i = np.argmin([r['mdl'][-1] for r in results])
    return best_model_i

models, models_results = get_models_results(models_dir)

###########################################################################

model_i = 2
best_model_i = [get_best_model_i(models_results[model_i][dataset_size]) for dataset_size in dataset_sizes]
best_models = [models[model_i][dataset_size][best_model_i[i]] for i, dataset_size in enumerate(dataset_sizes)]


# Write models and data to files
for model_i in range(n_models):

    with open(f'{groundtruth_models_dir}model_{model_i}.pkl', 'rb') as f:
        groundtruth_compressor = pickle.load(f)
    
    # Write nonterminals to file
    with open(f'{figs_dir}nonterminals_gt_{model_i}.txt', 'w') as f:
        nonterminals = [x for x in groundtruth_compressor.library.values() if x.type == 'nonterminal']
        # Sort them by node order
        nonterminals = sorted(nonterminals, key=lambda x: x.order)
        for n in nonterminals:
            f.write(f'{n}\n')

    data = groundtruth_compressor.generate_dataset(size=10)
    with open(f'{figs_dir}data_gt_{model_i}.txt', 'w') as f:
        for x in data:
            f.write(f'{x}\n')

    best_model_i = [get_best_model_i(models_results[model_i][dataset_size]) for dataset_size in dataset_sizes]
    best_models = [models[model_i][dataset_size][best_model_i[i]] for i, dataset_size in enumerate(dataset_sizes)]

    # Write nonterminals to file
    with open(f'{figs_dir}nonterminals_fitted_{model_i}.txt', 'w') as f:
        compressor = best_models[1]
        nonterminals = [x for x in compressor.library.values() if x.type == 'nonterminal']
        # Sort them by order
        nonterminals = sorted(nonterminals, key=lambda x: x.order)
        for n in nonterminals:
            f.write(f'{n}\n')

    test_data = best_models[1].generate_dataset(size=10)
    with open(f'{figs_dir}data_fitted_{model_i}.txt', 'w') as f:
        for x in test_data:
            f.write(f'{x}\n')    

# Plot libraries for each model
for model_i in range(n_models):

    with open(f'{groundtruth_models_dir}model_{model_i}.pkl', 'rb') as f:
        groundtruth_compressor = pickle.load(f)

    groundtruth_library_mdl = groundtruth_compressor.get_entropy()
    best_model_i = [get_best_model_i(models_results[model_i][dataset_size]) for dataset_size in dataset_sizes]
    best_models = [models[model_i][dataset_size][best_model_i[i]] for i, dataset_size in enumerate(dataset_sizes)]
    fitted_library_mdl = [compressor.get_entropy() for compressor in best_models]

    groundtruth_compressor.plot(save_name=f'{figs_dir}gt_library_model_{model_i}.png')
    best_models[1].plot(save_name=f'{figs_dir}fitted_library_model_{model_i}.png')

# Plot MDL as a function of dataset size
f, ax = plt.subplots(1, n_models, figsize=(3*n_models, 3), sharex=True, sharey=True)
for model_i in range(n_models):

    with open(f'{groundtruth_models_dir}model_{model_i}.pkl', 'rb') as f:
        groundtruth_compressor = pickle.load(f)

    groundtruth_library_mdl = groundtruth_compressor.get_entropy()
    best_model_i = [get_best_model_i(models_results[model_i][dataset_size]) for dataset_size in dataset_sizes]
    best_models = [models[model_i][dataset_size][best_model_i[i]] for i, dataset_size in enumerate(dataset_sizes)]
    fitted_library_mdl = [compressor.get_entropy() for compressor in best_models]
    
    n_test_data = 100
    groundtruth_data_mdl = []
    fitted_data_mdl = []
    for _ in range(n_test_data):
        print(f'Computing MDL on test data: {_}')
        test_data = groundtruth_compressor.generate_dataset(size=100)
        test_data_n = len(flatten(test_data))
        groundtruth_data_mdl.append(-groundtruth_compressor.get_dataset_ll(test_data)/test_data_n)
        fitted_data_mdl.append([-compressor.get_dataset_ll(test_data)/test_data_n for compressor in best_models])
    fitted_data_mdl = np.array(fitted_data_mdl)
    groundtruth_data_mdl_mean = np.mean(groundtruth_data_mdl, axis=0)
        
    # Fitted
    fitted_data_mdl_mean = np.mean(fitted_data_mdl, axis=0)
    ax[model_i].plot(range(len(fitted_data_mdl_mean)), fitted_data_mdl_mean, color='k', lw=3, ls='--',)
    for i, dataset_size in enumerate(dataset_sizes):
        ax[model_i].errorbar(range(len(fitted_data_mdl_mean)), fitted_data_mdl_mean, yerr=np.std(np.array(fitted_data_mdl), axis=0), fmt='o', color='k', lw=3, capsize=5)
    # Ground truth line
    ax[model_i].plot(range(len(fitted_data_mdl_mean)), [groundtruth_data_mdl_mean]*len(fitted_data_mdl_mean), color='k', lw=3)
    ax[model_i].errorbar(range(len(fitted_data_mdl_mean)), [groundtruth_data_mdl_mean]*len(fitted_data_mdl_mean), yerr=np.std(groundtruth_data_mdl), fmt='o', color='k', lw=3, capsize=5)

    # Remove top spine
    ax[model_i].spines['top'].set_visible(False)
    ax[model_i].spines['right'].set_visible(False)

    ax[model_i].set_xlabel('dataset size')

    ax[model_i].set_title(f'Grammar {model_i+1}')
    
ax[0].set_xticks(range(len(dataset_sizes)))
ax[0].set_xticklabels(dataset_sizes)
ax[0].set_yticks([0, 3])
ax[0].set_ylim(0, 3)
ax[0].set_xlim(-1, len(dataset_sizes))
ax[0].set_ylabel('test data cost\n(bits/syllable)')
plt.tight_layout()
plt.savefig(f'{figs_dir}mdl_results.png', dpi=300)
plt.savefig(f'{figs_dir}mdl_results.pdf', dpi=300)
plt.close()