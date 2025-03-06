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

compressor = ProbZip(alpha=1)
triplets     = ['abc', 'def', 'ghi', 'jkl', 'mno']
dataset       = [''.join(np.random.choice(triplets, 50)) for _ in range(100)]
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=42)

results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                      dataset_val=dataset_val,
                                      steps=1000,
                                      log_every=100)

symbol = compressor.compress(dataset_test[0], update_counts=False)
print(symbol)
print(symbol.probability_compress(dataset_test[0], 0))
print(symbol.probability_not_compress(dataset_test[0], 0))
print(symbol.probability(dataset_test[0], 0))

plt.plot(ll_dict['train'], label='train')
plt.plot(ll_dict['val'], label='val')
plt.legend()
plt.show()

compressor = ProbZip(alpha=1)
compressor.get_terminals(flatten_arbitrarily_nested_lists(dataset))
compressor.library["['a', 'b']"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['b'], rate=None)
compressor.library["[['a', 'b'], 'c']"] = Node(alpha=compressor.alpha, parent=compressor.library["['a', 'b']"], suffix=compressor.library['c'], rate=None)

compressor.library["['d', 'e']"] = Node(alpha=compressor.alpha, parent=compressor.library['d'], suffix=compressor.library['e'], rate=None)
compressor.library["[['d', 'e'], 'f']"] = Node(alpha=compressor.alpha, parent=compressor.library["['d', 'e']"], suffix=compressor.library['f'], rate=None)

compressor.library["['g', 'h']"] = Node(alpha=compressor.alpha, parent=compressor.library['g'], suffix=compressor.library['h'], rate=None)
compressor.library["[['g', 'h'], 'i']"] = Node(alpha=compressor.alpha, parent=compressor.library["['g', 'h']"], suffix=compressor.library['i'], rate=None)

compressor.library["['j', 'k']"] = Node(alpha=compressor.alpha, parent=compressor.library['j'], suffix=compressor.library['k'], rate=None)
compressor.library["[['j', 'k'], 'l']"] = Node(alpha=compressor.alpha, parent=compressor.library["['j', 'k']"], suffix=compressor.library['l'], rate=None)

compressor.library["['m', 'n']"] = Node(alpha=compressor.alpha, parent=compressor.library['m'], suffix=compressor.library['n'], rate=None)
compressor.library["[['m', 'n'], 'o']"] = Node(alpha=compressor.alpha, parent=compressor.library["['m', 'n']"], suffix=compressor.library['o'], rate=None)

# compressor.library['x'] = Node(alpha=compressor.alpha, parent=compressor.epsilon, suffix='x', rate=None)
# data_test = 'abcdefghijklmnoxxx'
symbol = compressor.compress(data_test, update_counts=False)
print(symbol)
print(symbol.probability_compress(data_test, 0))
print(symbol.probability_not_compress(data_test, 0))
print(symbol.probability(data_test, 0))

################################################################

data = get_data(experimenter='Lena', strings=True)
subject_dataset = data['wh09pk88']
dataset_train, dataset_test = train_test_split(subject_dataset, test_size=0.2, random_state=42)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=42)

d = {}

for alpha in alphas:

    print('-----------------------------------')
    print(f'alpha: {alpha}')

    # ProbZip
    compressor = ProbZip(alpha=alpha)
    results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                                dataset_val=dataset_val,
                                                steps=10000,
                                                log_every=1000)
    d[alpha] = results_dict

f, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=False)
for alpha, results_dict in d.items():
    ax[0].plot(results_dict['ll_train'], label=alpha)
    ax[1].plot(results_dict['ll_val'], label=alpha)
    ax[2].plot(results_dict['entropy'], label=alpha)
ax[2].legend(title='alpha')
ax[0].set_ylabel('ll train')
ax[1].set_ylabel('ll validation')
ax[2].set_ylabel('library entropy')
ax[0].set_xlabel('iter')
ax[1].set_xlabel('iter')
ax[2].set_xlabel('iter')
ax[0].set_xticks(np.arange(0, 11, 5))
ax[0].set_xticklabels(np.arange(0, 10001, 5000))
ax[1].set_xticklabels(np.arange(0, 10001, 5000))
ax[2].set_xticklabels(np.arange(0, 10001, 5000))
plt.tight_layout()
plt.show()

################################################################

compressor = ProbZip(alpha=.001)
results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                            dataset_val=dataset_val,
                                            steps=10000,
                                            log_every=1000)
print(len(compressor.library))
data_test = dataset_test[0]
symbol = compressor.compress(data_test, update_counts=False)
print(symbol)
print(symbol.probability_compress(data_test, 0))
print(symbol.probability_not_compress(data_test, 0))
print(symbol.probability(data_test, 0))

f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(results_dict['ll_train'], label='train')
ax[0].plot(results_dict['ll_val'], label='val')
ax[0].legend()
ax[1].plot(results_dict['entropy'], label='entropy')
ax[1].legend()
plt.show()


################################################################

data = get_data(experimenter='Lena', strings=True)
subjects = list(data.keys())

subject_dataset = data['wh09pk88']
dataset_train, dataset_test = train_test_split(subject_dataset, test_size=0.2, random_state=42)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=42)

# import time

compressor = ProbZip(alpha=1)

results_dict = compressor.compress_dataset(dataset_train=dataset_train,
                                      dataset_val=dataset_val,
                                      steps=100000,
                                      log_every=1000)

symbol = compressor.compress(dataset_test[0], update_counts=False)
print(symbol)
print(symbol.probability_compress(dataset_test[0], 0))
print(symbol.probability_not_compress(dataset_test[0], 0))
print(symbol.probability(dataset_test[0], 0))

len(compressor.library)
compressor.get_dataset_shannon_entropy(dataset_test)


f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(results_dict['ll_train'], label='train')
ax[0].plot(results_dict['ll_val'], label='val')
ax[0].legend()
ax[1].plot(results_dict['entropy'], label='entropy')
ax[1].legend()
plt.show()

d = []
for subject in ['wh09pk88']:

    print(f'Working on subject {subject}')

    subject_dataset = data[subject]

    kf = KFold(n_splits=k)
    for fold, (train_index, test_index) in enumerate(kf.split(subject_dataset)):

        print(f'Working on fold {fold}')

        dataset_train, dataset_test = np.array(subject_dataset)[train_index], np.array(subject_dataset)[test_index]
        dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=42)

        for alpha in alphas:
            
            print(f'alpha: {alpha}')

            # ProbZip
            compressor = ProbZip(alpha=alpha)
            _ = compressor.compress_dataset(dataset_train=dataset_train, dataset_val=dataset_val, steps=10000, log_every=10000)
            ll_train = compressor.get_dataset_ll(dataset_test)
            ll_test = compressor.get_dataset_ll(dataset_test)
            entropy = compressor.get_shannon_entropy()

            d.append([subject, fold, alpha, ll_train, ll_test, entropy])

            del compressor
            gc.collect()  # Force garbage collection

        del dataset_train, dataset_test, dataset_val
        gc.collect()
        
df = pd.DataFrame(d, columns=['subject', 'fold', 'alpha', 'll_train', 'll_test', 'entropy'])
# Get best_d

d.columns = ['subject', 'fold', 'alpha', 'll_train', 'll_test', 'entropy_train', 'entropy_test']
best_d.columns = ['subject', 'fold', 'alpha', 'll_train', 'll_test', 'entropy_train', 'entropy_test']
d.to_csv('ll_by_alpha.csv')
best_d.to_csv('best_alpha.csv')

### Load and analyze data

# d = pd.read_csv('ll_by_alpha.csv')

sns.catplot(data=d, x='alpha', hue='subject', y='ll_test', sharey=False, sharex=False, kind='point', legend=False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# _, ax = plt.subplots(figsize=(20, 5))
# sns.barplot(data=d, x='subject', y='ll_test', ax=ax)
# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.tight_layout()
# plt.show()