from bengalese_finch.models.probzip import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

#################################################################################

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_new_new/'
figs_dir = 'bengalese_finch/analyses/probzip/figs/'
data = get_data(experimenter='Lena', strings=True)
subjects = list(data.keys())

def get_models_results(models_dir):
    # Construct the models and results dictionaries
    models = {}
    models_results = {}
    for subject in subjects:
        models[subject] = []
        models_results[subject] = []
        for model_i in range(10):
            try:
                with open(f'{models_dir}model_{subject}_{model_i}.pkl', 'rb') as f:
                    compressor = pickle.load(f)
            except FileNotFoundError:
                print(f'Model file not found for subject {subject}, model {model_i}. Skipping.')
                continue
            try:
                with open(f'{models_dir}results_{subject}_{model_i}.pkl', 'rb') as f:
                    results_dict = pickle.load(f)
            except FileNotFoundError:
                print(f'Results file not found for subject {subject}, model {model_i}. Skipping.')
                continue

            models[subject].append(compressor)
            models_results[subject].append(results_dict)

    return models, models_results

def plot_1(models_results): 

    f, ax = plt.subplots(1, 2, figsize=(8, 4))

    for subject_i, subject in enumerate(models_results.keys()):
        best_model_i = get_best_model_i(models_results[subject])
        results_dict = models_results[subject][best_model_i]

        ax[0].plot(results_dict['mdl_data_test'], label=f'{subject}', lw=2, alpha=1)
        ax[1].plot(results_dict['mdl_library'], label=f'{subject}', lw=2, alpha=1)

    # Add average line; Ignore np.inf values; Do it in a loop
    mdl_data_test_all = []
    mdl_library_all = []
    for subject in models_results.keys():
        best_model_i = get_best_model_i(models_results[subject])
        if not np.isinf(models_results[subject][best_model_i]['mdl_data_test']).any():
            mdl_data_test_all.append(models_results[subject][best_model_i]['mdl_data_test'])
        if not np.isinf(models_results[subject][best_model_i]['mdl_library']).any():
            mdl_library_all.append(models_results[subject][best_model_i]['mdl_library'])
    mdl_data_test_all = np.mean(mdl_data_test_all, axis=0)
    mdl_library_all = np.mean(mdl_library_all, axis=0)

    ax[0].plot(mdl_data_test_all, color='k', lw=4, alpha=1)
    ax[1].plot(mdl_library_all, color='k', lw=4, alpha=1)
    # ax[2].plot(mdl_all, color='k', lw=4, alpha=1)

    ax[0].set_title('test data')
    ax[1].set_title('ProbZip library')
    ax[0].set_ylabel('bits/syllable')
    # Legend on the right
    ax[1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

    ax[0].set_ylim(0, 10)
    ax[1].set_ylim(0, .005)
    ax[0].set_yticks([0, 10])
    ax[1].set_yticks([0, .005])

    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('step')
        ax[i].set_xticklabels(['0', '100k'])
        ax[i].set_xticks(range(0, 101, 100))

    plt.tight_layout()
    # Save both in png and pdf formats
    plt.savefig(f'{figs_dir}baseline_result.png', dpi=300)
    plt.savefig(f'{figs_dir}baseline_result.pdf', dpi=300)
    plt.close('all')

def plot_2(models_results, subject):
    """
    Plot the results of the models.
    """
    m_r = models_results[subject]
    f, ax = plt.subplots(1, 3, figsize=(8, 4))
    for model_i, results_dict in enumerate(m_r):
        ax[0].plot(results_dict['mdl_data_test'], label=f'seed {model_i}', lw=4, alpha=1)
    for model_i, results_dict in enumerate(m_r):
        ax[1].plot(results_dict['mdl_library'], label=f'seed {model_i}', lw=4, alpha=1)
    for model_i, results_dict in enumerate(m_r):
        ax[2].plot(results_dict['mdl'], label=f'seed {model_i}', lw=4, alpha=1)
    # ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].set_title('MDL test data')
    ax[1].set_title('MDL library')
    ax[2].set_title('MDL')

    for i in range(3):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('step')
        ax[i].set_xticklabels(['0', '100k'])
        ax[i].set_xticks(range(0, 101, 100))

    plt.tight_layout()
    # Save both in png and pdf formats
    plt.savefig(f'{figs_dir}baseline_result_{subject}.png', dpi=300)
    plt.savefig(f'{figs_dir}baseline_result_{subject}.pdf', dpi=300)
    plt.close('all')

# TODO(noemielteto): Goes in analysis library
def get_best_model_i(results):
    best_model_i = np.argmin([r['mdl'][-1] for r in results])
    return best_model_i

###########################################################################


models, models_results = get_models_results(models_dir)

# plot_1(models_results)

for subject in subjects:

    plot_2(models_results, subject)

    best_model_i = get_best_model_i(models_results[subject])
    best_model = models[subject][best_model_i]
    # best_model.plot(save_name=f'{figs_dir}baseline_library_{subject}.png')
    # best_model.write_to_txt(f'{figs_dir}baseline_library_{subject}.txt')

    subject_dataset = data[subject]

    _, dataset_test = train_test_split(subject_dataset,
                                                    test_size=0.2,
                                                    random_state=best_model_i+10)
    _, dataset_test = train_test_split(dataset_test,
                                                    test_size=0.5,
                                                    random_state=best_model_i+10)
    
    test_bout = dataset_test[0]
    symbol = best_model.compress(test_bout)
    # Write test bout and parse to txt
    with open(f'{figs_dir}baseline_test_parse_{subject}.txt', 'w') as f:
        f.write(f'Subject: {subject}\n')
        f.write(f'Raw bout: {test_bout}\n') 
        f.write(f'Parsed bout: {symbol.expression}\n')