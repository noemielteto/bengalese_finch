from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

alpha_vector = [1]*1000
compressor = ProbZip(alpha_vector=alpha_vector)

data = get_data(experimenter='Lena', strings=True)
subject_dataset = data['wh09pk88']
dataset_train, dataset_test = train_test_split(subject_dataset,
                                               test_size=0.2,
                                               random_state=42)
dataset_val, dataset_test = train_test_split(dataset_test,
                                             test_size=0.5,
                                             random_state=42)

results_dict = compressor.search_add_remove(dataset_train=dataset_train,
                                            dataset_val=dataset_val,
                                            steps=10,
                                            log_every=1)