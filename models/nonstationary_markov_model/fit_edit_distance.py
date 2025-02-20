from utils import *
from nonstationary_markov_model import *

def fit(subject):

    bouts = data[subject]
    bouts = bouts[:int(len(bouts)*.8)]  # training data will be first 80%

    E, I, S = get_E_I_S(bouts)

    # d = np.zeros((len(decay_rates), len(restore_rates), len(s_thresholds)))
    d = pd.DataFrame()
    for i, s_threshold in enumerate(s_thresholds):
        print(i)
        for j, decay_rate_value in enumerate(decay_rates):
            print(j)
            for k, restore_rate_value in enumerate(restore_rates):
                print(k)
                print('---')
                theta = (decay_rate_value, restore_rate_value, s_threshold)
                edit_distance = NSMM_edit_distance_allbouts_knownmatrix(theta, E, I, S, bouts)
                # d[i][j][k] = edit_distance
                d = d.append(pd.Series([subject, s_threshold, decay_rate_value, restore_rate_value, edit_distance]), ignore_index=True)
    return d

data = get_data()
subjects = ['gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']
s_thresholds = np.linspace(0.05, 0.3, 15)
decay_rates = np.linspace(0.05, 0.5, 15)
restore_rates = np.linspace(0.0001, 0.05, 15)

import multiprocessing as mp
if __name__ == '__main__':
    # n_physical_cores = psutil.cpu_count(logical = False)
    pool = mp.Pool(len(subjects))
    ds = pool.map(fit, subjects)
    pool.close()
    d = pd.concat(ds)
    d.columns = ['subject', 's_threshold', 'decay_rate', 'restore_rate', 'edit_distance']
    d.to_csv('gridsearch_values_2.csv')
