from multiprocessing import Pool
from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gc
import os

alpha = 100

def fit(groundtruth_model_i, dataset_size):

    with open(f'{groundtruth_models_dir}model_{groundtruth_model_i}.pkl', 'rb') as f:
        compressor = pickle.load(f)

    subject_dataset = compressor.generate_dataset(size=dataset_size)

    for model_i in range(10):

        print(f'Groundtruth model {groundtruth_model_i}; dataset size {dataset_size}; model: {model_i}')

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
                                                    prune_every=1000,
                                                    log=False)
        
        with open(f'{models_dir}model_{groundtruth_model_i}_{dataset_size}_{model_i}.pkl', 'wb') as f:
            pickle.dump(compressor, f)

        with open(f'{models_dir}results_{groundtruth_model_i}_{dataset_size}_{model_i}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)

        # Free up memory
        del compressor
        del dataset_train
        del dataset_val
        del dataset_test
        del results_dict
        gc.collect()  # Force garbage collection

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_synthetic/'
groundtruth_models_dir = 'bengalese_finch/analyses/probzip/groundtruth_models_synthetic/'
n_models = len([name for name in os.listdir(groundtruth_models_dir) if os.path.isfile(os.path.join(groundtruth_models_dir, name))])
dataset_sizes = [10, 100, 1000]
# dataset_sizes = [100]
# tasks = [(i, size) for i in range(n_models) for size in dataset_sizes]
tasks = [(i, size) for i in [2] for size in dataset_sizes]

# Parallelize the fitting process
if __name__ == "__main__":

    # Use pool.starmap to parallelize the tasks
    with Pool() as pool:
        pool.starmap(fit, tasks)