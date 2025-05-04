from bengalese_finch.models.probzip import *
import pandas as pd
import pickle

models_dir = 'bengalese_finch/analyses/probzip/fitted_models/'
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

# TODO(noemielteto): Goes in analysis library
def get_best_model_i(results):
    best_model_i = np.argmin([r['mdl'][-1] for r in results])
    return best_model_i

###########################################################################

def get_targeted_syllables(subject):

    ########################### Lena sample ###########################

    if subject == 'bu86bu48':
        target = {'1': 'cx', '2':'cr'}
    elif subject == 'wh08pk40':
        target = {'1': 'ac', '2':'ec'}
    elif subject == 'wh09pk88':
        target = {'1': 'fab', '2':'nab'}
    elif subject == 'rd49rd79':
        target = {'1': 'ab', '2':'dx'}
    elif subject == 'gr57bu40':
        target = {'1': 'fcb', '2':'dd'}
    elif subject == 'gr58bu60':
        target = {'1': 'nab', '2':'nad'}
    elif subject == 'gr54bu78':
        target = {'1': 'ee', '2':'ea'}
    elif subject == 'rd82wh13':
        target = {'1': 'ld', '2':'le'}

    # ########################### Simon sample ###########################
    # elif subject == 'ye0wh0':
    #     s = ['a', 'b', 'e']
    # elif subject == 'rd6bu6':
    #     s = ['c', 'j', 'k']
    # elif subject == 'rd6030':
    #     s = ['b', 'h', 'y']
    # elif subject == 'rd8031':
    #     s = ['a', 'g', 'l']
    # elif subject == 'rd5374':
    #     s = ['b', 'j', 'u']

    return target
    
def get_metrics_of_targeted_syllables(compressor, subject):

    target = get_targeted_syllables(subject)
    orders = {'1': [], '2': []}
    hierarchies = {'1': [], '2': []}

    for t in target.keys():
        for s in target[t]:
            syllable_orders = []
            syllable_hierarchies = []
            for symbol in compressor.library.values():
                if s in symbol.flat_expression:
                    syllable_orders.append(symbol.order)
                    syllable_hierarchies.append(symbol.get_hierarchy())
            orders[t].append(max(syllable_orders))
            hierarchies[t].append(max(syllable_hierarchies))
    metrics = { '1': {'order': np.mean(orders['1']),
                        'hierarchy': np.mean(hierarchies['1'])},
                '2': {'order': np.mean(orders['2']),
                        'hierarchy': np.mean(hierarchies['2'])},
    }

    return metrics

def get_learning_scores(scores_dir='bengalese_finch/analyses/probzip/Lena_learning_scores.csv'):

    d = pd.read_csv(scores_dir, sep=';', index_col=0)
    d = d.T
    # Create a MultiIndex for the columns
    multi_index = pd.MultiIndex.from_tuples(
        [('baseline', 'C1'), ('baseline', 'C2'), ('training', 'C1'), ('training', 'C2')],
        names=['phase', 'context']
    )
    d.columns = multi_index

    # how much T1 is suppressed in C1 (where T1 is punished)
    d[('change_score', 'C1')] =  d[('baseline', 'C1')] - d[('training', 'C1')]
    # how much T1 is enhanced in C2 (where T1 is rewarded)
    d[('change_score', 'C2')] = d[('training', 'C2')] - d[('baseline', 'C2')]

    d['cue_difference_score'] =  d[('training', 'C2')] - d[('training', 'C1')]

    d['average_change_score'] = (d[('change_score', 'C1')] + d[('change_score', 'C2')]) / 2

    return d

models, models_results = get_models_results(models_dir)
learning_scores = get_learning_scores()
# Analyse posttrain_difference_score only -- which is the same for both contexts since it's their absolute difference
learning_scores[('order', 'T1')] = np.nan
learning_scores[('order', 'T2')] = np.nan
learning_scores[('hierarchy', 'T1')] = np.nan
learning_scores[('hierarchy', 'T2')] = np.nan

for subject in subjects:

    best_model_i = get_best_model_i(models_results[subject])
    best_model = models[subject][best_model_i]
    metrics = get_metrics_of_targeted_syllables(best_model, subject)
    learning_scores.at[subject, ('order', 'T1')] = metrics['1']['order']
    learning_scores.at[subject, ('order', 'T2')] = metrics['2']['order']
    learning_scores.at[subject, ('hierarchy', 'T1')] = metrics['1']['hierarchy']
    learning_scores.at[subject, ('hierarchy', 'T2')] = metrics['2']['hierarchy']

learning_scores['average_order'] = (learning_scores[('order', 'T1')] + learning_scores[('order', 'T2')]) / 2
learning_scores['average_hierarchy'] = (learning_scores[('hierarchy', 'T1')] + learning_scores[('hierarchy', 'T2')]) / 2

# Flatten the MultiIndex columns
learning_scores.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in learning_scores.columns]

plt.scatter(learning_scores['cue_difference_score_'], learning_scores['average_hierarchy_'])
plt.tight_layout()
plt.show()

sns.regplot(data=learning_scores, x='change_score_C2', y='hierarchy_T1')
plt.tight_layout()
plt.show()

# Spearman correlation
from scipy.stats import spearmanr
spearmanr(learning_scores['change_score_C1'], learning_scores['hierarchy_T1'])
spearmanr(learning_scores['change_score_C2'], learning_scores['hierarchy_T1'])
spearmanr(learning_scores['change_score_C1'], learning_scores['hierarchy_T2'])
spearmanr(learning_scores['change_score_C2'], learning_scores['hierarchy_T2'])

spearmanr(learning_scores['cue_difference_score_'], learning_scores['average_hierarchy_'])

x = learning_scores[learning_scores.index != 'wh08pk40']
spearmanr(x['cue_difference_score_'], x['average_hierarchy_'])