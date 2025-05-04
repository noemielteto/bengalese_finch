from concurrent.futures import ProcessPoolExecutor
from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import gc

models_dir = 'bengalese_finch/analyses/probzip/fitted_models/'
alpha = 100

def lengthmatch_data(data, n=50, tol=5):
    """
    Subsample the data to match the song bout length between pre and post lesion.
    If the number of song bouts is less than n, it will not subsample.
    """
    matched_data = {}
    for subject in data:
        matched_data[subject] = {'prelesion': [], 'postlesion': []}
        # For each song in prelesion, find the closest song in postlesion
        for song in data[subject]['prelesion']:
            # Get the length of the song
            song_length = len(song)
            # Find the closest song in postlesion
            closest_song = min(data[subject]['postlesion'], key=lambda x: abs(len(x) - song_length))
            # Check if the difference is within the tolerance
            if abs(len(closest_song) - song_length) < tol:
                matched_data[subject]['prelesion'].append(song)
                matched_data[subject]['postlesion'].append(closest_song)
        # If the number of song bouts is less than n, throw error
        if len(matched_data[subject]['prelesion']) < n:
            # Error message
            print(f'Not enough song bouts for subject {subject}. Prelesion: {len(matched_data[subject]["prelesion"])}, Postlesion: {len(matched_data[subject]["postlesion"])}. Try increasing tol.')
            break
    
    return matched_data

def subsample_data(data):
    """
    Subsample data to match the number of song bouts among subjects.
    """
    min_bouts = min([len(data[subject]['prelesion']) for subject in data])
    for subject in data:
        data[subject]['prelesion'] = np.random.choice(data[subject]['prelesion'], min_bouts, replace=False).tolist()
        data[subject]['postlesion'] = np.random.choice(data[subject]['postlesion'], min_bouts, replace=False).tolist()
    return data

def fit(subject):
    """
    Fit the ProbZip model to the data for a given subject.
    """
    for phase in ['prelesion', 'postlesion']:

        subject_dataset = data[subject][phase]

        for model_i in range(10):

            print(f'Subject {subject}; Phase {phase}; Model: {model_i}')

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
            
            with open(f'{models_dir}model_{subject}_{phase}_{model_i}_lenmatched.pkl', 'wb') as f:
                pickle.dump(compressor, f)

            with open(f'{models_dir}results_{subject}_{phase}_{model_i}_lenmatched.pkl', 'wb') as f:
                pickle.dump(results_dict, f)

            # Free up memory
            del compressor
            del dataset_train
            del dataset_val
            del dataset_test
            del results_dict
            gc.collect()  # Force garbage collection

data = get_data_lesion(strings=True)
data = lengthmatch_data(data, n=50, tol=5)
data = subsample_data(data)
subjects = list(data.keys())

if __name__ == "__main__":
    # Use ProcessPoolExecutor to fit all subjects in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(fit, subjects)