from concurrent.futures import ThreadPoolExecutor
from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gc

alpha = 100

#################################################################################

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_new_new/'
data = get_data(experimenter='Lena', strings=True)
subjects = list(data.keys())
# min_len_data = min([len(v) for v in data.values()])
min_len_data = 100

def fit(subject, alpha=alpha):
    """
    Fit the ProbZip model to the data for a given subject.
    """
    subject_dataset = data[subject][:min_len_data]
    for model_i in range(10):

        print(f'Subject {subject}; model: {model_i}')

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
                                                    steps=10000,
                                                    prune_every=1000,
                                                    log_every=10000)
        
        with open(f'{models_dir}model_{subject}_{model_i}.pkl', 'wb') as f:
            pickle.dump(compressor, f)

        with open(f'{models_dir}results_{subject}_{model_i}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)

        # Free up memory
        del compressor
        del dataset_train
        del dataset_val
        del dataset_test
        del results_dict
        gc.collect()  # Force garbage collection

# Parallelize the fitting process
if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        executor.map(fit, subjects)