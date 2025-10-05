from multiprocessing import Pool
from bengalese_finch.models.probzip import *
from sklearn.model_selection import train_test_split
import pickle
import gc
import os

alpha = 10
n_val = 10
n_test = 10

def fit(groundtruth_model_i, dataset_size):

    with open(f'{groundtruth_models_dir}model_{groundtruth_model_i}.pkl', 'rb') as f:
        compressor = pickle.load(f)

    # Validation and tes sets are of size 10; training set is of size dataset_size
    dataset = compressor.generate_dataset(size=dataset_size+n_val+n_test)

    for model_i in range(10):

        print(f'Groundtruth model {groundtruth_model_i}; dataset size {dataset_size}; model: {model_i}')

        dataset_shuffled = np.random.choice(dataset, size=dataset_size+n_val+n_test, replace=False).tolist()
        dataset_train = dataset_shuffled[:dataset_size]
        dataset_val = dataset_shuffled[dataset_size:dataset_size+n_val]
        dataset_test = dataset_shuffled[dataset_size+n_val:dataset_size+n_val+n_test]

        compressor = ProbZip(alpha=alpha)
        results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                                    dataset_val=dataset_val,
                                                    dataset_test=dataset_test,
                                                    steps=10000,
                                                    prune_every=100,
                                                    log=True,
                                                    log_every=1000)
        
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

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_synthetic_new/'
groundtruth_models_dir = 'bengalese_finch/analyses/probzip/groundtruth_models_synthetic/'
n_models = len([name for name in os.listdir(groundtruth_models_dir) if os.path.isfile(os.path.join(groundtruth_models_dir, name))])
dataset_sizes = [10, 50, 100]
# dataset_sizes = [100]
# tasks = [(i, size) for i in range(n_models) for size in dataset_sizes]
tasks = [(i, size) for i in [1] for size in [50]]
# tasks = [(i, size) for i in [0] for size in dataset_sizes]

# Parallelize the fitting process
if __name__ == "__main__":

    # Use pool.starmap to parallelize the tasks
    with Pool() as pool:
        pool.starmap(fit, tasks)