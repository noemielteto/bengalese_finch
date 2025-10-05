from multiprocessing import Pool
from bengalese_finch.models.probzip import *
import pickle
import gc
import pretty_midi

midi_data = pretty_midi.PrettyMIDI("bach_846.mid")
instrument = midi_data.instruments[0]  # often piano

# Get readable note names with timing
data = []
for note in instrument.notes:
     data.append(pretty_midi.note_number_to_name(note.pitch))
data = ['<'] + data + ['>']

alpha = 10

def fit(model_i):

        compressor = ProbZip(alpha=alpha)

        dataset_train = data
        dataset_test = dataset_train
        dataset_val = dataset_train

        results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                                    dataset_val=dataset_val,
                                                    dataset_test=dataset_test,
                                                    steps=10000,
                                                    prune_every=1000,
                                                    log=True,
                                                    log_every=100)
        
        with open(f'{models_dir}model_{model_i}.pkl', 'wb') as f:
            pickle.dump(compressor, f)

        with open(f'{models_dir}results_{model_i}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)

        # Free up memory
        del compressor
        del dataset_train
        del dataset_val
        del dataset_test
        del results_dict
        gc.collect()  # Force garbage collection

models_dir = 'bengalese_finch/analyses/probzip/fitted_models_music/'
runs = [i for i in range(8)]

# Parallelize the fitting process
if __name__ == "__main__":

    # Use pool.starmap to parallelize the tasks
    with Pool() as pool:
        pool.map(fit, runs)