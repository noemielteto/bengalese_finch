from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import train_test_split

HCRP_alphas = [1, 10, 100, 1000, 10000, 100000, 1000000]
probzip_alphas = [.0000001, .000001, .00001, .0001, 0.001, 0.01, 0.1, 1]

# HCRP_alphas = [1, 10, 100]
# probzip_alphas = [.0000001]

data = get_data(experimenter='Lena', strings=True)

subjects = list(data.keys())

d = pd.DataFrame()
best_d = pd.DataFrame()

for subject in subjects:

    print(f'Working on subject {subject}')

    subject_dataset = data[subject]
    dataset_train, dataset_test = train_test_split(subject_dataset, test_size=0.2, random_state=42)

    # HCRP_ll_train, HCRP_ll_test = [], []
    probzip_ll_train, probzip_ll_test = [], []
    probzip_entropy_train, probzip_entropy_test = [], []
    # for alpha in HCRP_alphas:

    #     print(f'HCRP alpha: {alpha}')

    #     # HCRP
    #     m = HCRP_LM(strength=[alpha]*25)
    #     X_train = [list(x) for x in dataset_train]
    #     X_test = [list(x) for x in dataset_test]
    #     m.fit(X=X_train, frozen=True)
    #     ll_train = np.sum(np.log(m.likelihoods))
    #     HCRP_ll_train.append(ll_train)
    #     m.predict(X=X_test)
    #     ll_test = np.sum(np.log(m.likelihoods))
    #     HCRP_ll_test.append(ll_test)

    #     new_row = [subject, alpha, 'HCRP', ll_train, ll_test]
    #     d = pd.concat([d, pd.DataFrame([new_row])], ignore_index=True)

    for alpha in probzip_alphas:
        
        print(f'probzip alpha: {alpha}')

        # ProbZip
        compressor = ProbZip(alpha=alpha)
        compressor.compress_dataset(dataset=dataset_train, steps=1000)
        ll_train = compressor.get_dataset_ll(dataset_test)
        ll_test = compressor.get_dataset_ll(dataset_test)
        probzip_ll_train.append(ll_train)
        probzip_ll_test.append(ll_test)
        probzip_entropy_train.append(compressor.get_dataset_shannon_entropy(dataset_train))
        probzip_entropy_test.append(compressor.get_dataset_shannon_entropy(dataset_test))

        new_row = [subject, alpha, 'probzip', ll_train, ll_test, probzip_entropy_train, probzip_entropy_test]
        d = pd.concat([d, pd.DataFrame([new_row])], ignore_index=True)

    # HCRP_lik_test = np.max(HCRP_ll_test)
    # HCRP_lik_train = HCRP_ll_train[np.argmax(HCRP_ll_test)]
    # HCRP_best_alpha = HCRP_alphas[np.argmax(HCRP_ll_test)]
    # probzip_lik_test = np.max(probzip_ll_test)
    # probzip_lik_train = probzip_ll_train[np.argmax(probzip_ll_test)]
    # probzip_best_alpha = probzip_alphas[np.argmax(probzip_ll_test)]

    probzip_entropy_test = np.min(probzip_entropy_test)
    probzip_entropy_train = probzip_entropy_train[np.argmin(probzip_entropy_test)]
    probzip_best_alpha = probzip_alphas[np.argmin(probzip_entropy_test)]

    # new_row = [subject, HCRP_best_alpha, 'HCRP', HCRP_lik_train, HCRP_lik_test]
    # best_d = pd.concat([best_d, pd.DataFrame([new_row])], ignore_index=True)
    new_row = [subject, probzip_best_alpha, 'probzip', probzip_entropy_train, probzip_entropy_test]
    best_d = pd.concat([best_d, pd.DataFrame([new_row])], ignore_index=True)

d.to_csv('entropies_alphas.csv')
best_d.to_csv('entropies_optimized_alphas.csv')

### Load and analyze data

d = pd.read_csv('liks_optimized_alphas.csv')
d.columns = ['index', 'subject', 'alpha', 'model', 'll_train', 'll_test']

sns.catplot(data=d, x='alpha', hue='subject', col='model', y='ll_test', sharey=False, sharex=False, kind='point', legend=False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

_, ax = plt.subplots(figsize=(20, 5))
sns.barplot(data=d, x='subject', hue='model', y='ll_test', ax=ax)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
plt.show()