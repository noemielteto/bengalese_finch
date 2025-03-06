from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

alphas = [.0000001, .000001, .00001, .0001, 0.001,]

data = get_data_lesion(strings=True)
subjects = list(data.keys())

d = pd.DataFrame()
best_d = pd.DataFrame()

for subject in subjects:

    print(f'Working on subject {subject}')

    for phase in ['prelesion', 'postlesion']:

        print(f'Working on phase {phase}')

        subject_dataset = data[subject][phase]
        dataset_train, dataset_test = train_test_split(subject_dataset, test_size=0.2, random_state=42)

        probzip_ll_train, probzip_ll_test = [], []

        for alpha in alphas:
            
            print(f'probzip alpha: {alpha}')

            # ProbZip
            compressor = ProbZip(alpha=alpha)
            compressor.compress_dataset(dataset=dataset_train, steps=10000)
            ll_train = compressor.get_dataset_ll(dataset_test)
            ll_test = compressor.get_dataset_ll(dataset_test)
            probzip_ll_train.append(ll_train)
            probzip_ll_test.append(ll_test)

            new_row = [subject, phase, alpha, ll_train, ll_test]
            d = pd.concat([d, pd.DataFrame([new_row])], ignore_index=True)

        best_idx = np.argmax(probzip_ll_test)
        best_alpha = alphas[best_idx]

        new_row = [subject, phase, best_alpha, probzip_ll_train[best_idx], probzip_ll_test[best_idx]]
        best_d = pd.concat([best_d, pd.DataFrame([new_row])], ignore_index=True)

d.columns = ['subject', 'phase', 'alpha', 'll_train', 'll_test']
best_d.columns = ['subject', 'phase', 'alpha', 'll_train', 'll_test']

sns.catplot(data=d, x='alpha', hue='subject', col='phase', y='ll_test', sharey=False, sharex=False, kind='point', legend=False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

sns.catplot(data=best_d, x='subject', hue='phase', y='alpha', sharey=False, sharex=False, kind='bar', legend=False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()