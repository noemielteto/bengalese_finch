from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import gc

alphas = [1, 10, 100]
k = 3  # k-fold cross-validation

######################################

#################################################################################

alphas = [5, 10, 100]
models = {}
models_results = {}

for alpha in alphas:

    print(f'alpha: {alpha}')

    models[alpha] = []
    models_results[alpha] = []
    for model_i in range(3):

        print(f'Model: {model_i}')

        compressor = ProbZip(alpha=alpha)    

        dataset_train, dataset_test = train_test_split(subject_dataset,
                                                    test_size=0.2,
                                                    random_state=model_i+10)
        dataset_val, dataset_test = train_test_split(dataset_test,
                                                    test_size=0.5,
                                                    random_state=model_i+10)

        results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                                    dataset_val=dataset_val,
                                                    dataset_test=dataset_test,
                                                    steps=100000,
                                                    log_every=1000)
        
        models[alpha].append(compressor)
        models_results[alpha].append(results_dict)



# f, ax = plt.subplots(3, 3, sharey='col')
# for alpha_i, alpha in enumerate(models_results.keys()):
#     m_r = models_results[alpha]
#     for model_i, results_dict in enumerate(m_r):
#         ax[alpha_i][0].plot(results_dict['ll_val'], label=f'shuffle {model_i}', lw=4, alpha=1)
#     ax[alpha_i][0].set_ylabel('ll eval')
#     for model_i, results_dict in enumerate(m_r):
#         jittered_library_size = np.array(results_dict['library_size']) + np.random.uniform(0, 0.3, len(results_dict['library_size']))
#         ax[alpha_i][1].plot(jittered_library_size, label=f'shuffle {model_i}', lw=4, alpha=1)
#     ax[alpha_i][1].set_ylabel('library size')
#     for model_i, results_dict in enumerate(m_r):
#         ax[alpha_i][2].plot(results_dict['library_entropy'], label=f'shuffle {model_i}', lw=4, alpha=1)
#     ax[alpha_i][2].set_ylabel('library entropy')
#     ax[alpha_i][2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax[alpha_i][0].set_xticks(range(0, 101, 100))
#     ax[alpha_i][1].set_xticks(range(0, 101, 100))
#     ax[alpha_i][2].set_xticks(range(0, 101, 100))
#     ax[alpha_i][0].set_xticklabels(['0', '100k'])
#     ax[alpha_i][1].set_xticklabels(['0', '100k'])
#     ax[alpha_i][2].set_xticklabels(['0', '100k'])
#     ax[alpha_i][0].set_xlabel('step')
#     ax[alpha_i][1].set_xlabel('step')
#     ax[alpha_i][2].set_xlabel('step')
# plt.tight_layout()
# plt.show()

f, ax = plt.subplots(3, 3, sharey='col')
for alpha_i, alpha in enumerate(models_results.keys()):
    m_r = models_results[alpha]
    for model_i, results_dict in enumerate(m_r):
        ax[alpha_i][0].plot(results_dict['mdl_data_test'], label=f'seed {model_i}', lw=4, alpha=1)
    ax[alpha_i][0].set_ylabel('MDL test data')
    for model_i, results_dict in enumerate(m_r):
        ax[alpha_i][1].plot(results_dict['mdl_library'], label=f'seed {model_i}', lw=4, alpha=1)
    ax[alpha_i][1].set_ylabel('MDL library')
    for model_i, results_dict in enumerate(m_r):
        ax[alpha_i][2].plot(results_dict['mdl'], label=f'seed {model_i}', lw=4, alpha=1)
    ax[alpha_i][2].set_ylabel('MDL')
    ax[alpha_i][2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[alpha_i][0].set_xticks(range(0, 101, 100))
    ax[alpha_i][1].set_xticks(range(0, 101, 100))
    ax[alpha_i][2].set_xticks(range(0, 101, 100))
    ax[alpha_i][0].set_xticklabels(['0', '100k'])
    ax[alpha_i][1].set_xticklabels(['0', '100k'])
    ax[alpha_i][2].set_xticklabels(['0', '100k'])
    ax[alpha_i][0].set_xlabel('step')
    ax[alpha_i][1].set_xlabel('step')
    ax[alpha_i][2].set_xlabel('step')
# plt.tight_layout()
plt.show()