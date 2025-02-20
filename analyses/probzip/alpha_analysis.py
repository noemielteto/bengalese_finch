from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import train_test_split

data = get_data(experimenter='Lena', strings=True)

subject_dataset = data['bu86bu48']
# create a random train and test split
dataset_train, dataset_test = train_test_split(subject_dataset, test_size=0.2, random_state=42)

likelihoods_train, likelihoods_test = [], []
alphas = [.000001, .00001, .0001, 0.001, 0.01, 0.1, 1]
for alpha in alphas:
    print(f'alpha: {alpha}')
    compressor = ProbZip(alpha=alpha)
    compressor.compress_dataset(dataset=dataset_train, steps=1000)
    likelihood_train = compressor.get_dataset_ll(dataset_test)
    likelihood_test = compressor.get_dataset_ll(dataset_test)
    likelihoods_train.append(likelihood_train)
    likelihoods_test.append(likelihood_test)


plt.plot(likelihoods_train, c='k', label='train')
plt.plot(likelihoods_test, c='r', label='test')
plt.xticks(range(len(alphas)), alphas)
plt.legend()
plt.show()



######################################################


subjects = list(data.keys())

d = pd.DataFrame()

for subject in subjects:
    batch = data[subject]

    alphas = [1, 10, 100, 1000]
    size_library = []
    mean_expression_length = []
    for alpha in alphas:
        print(subject)
        print(alpha)
        compressor = ProbZip(alpha=alpha)
        compressor.compress_batch(batch)

        library_size = len(compressor.library)
        median_expression_length = np.median([len(node.flat_expression) for node in compressor.library.values()])

        compression_rates = []
        for x in range(len(batch)):
            if len(batch[x])==0:
                continue
            encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(batch[x])
            compression_rates.append(len(batch[x])/len(encoded_sequence))
        median_compression_rate = np.median(compression_rates)

        d = d.append(pd.Series([subject, alpha, library_size, median_expression_length, median_compression_rate]), ignore_index=True)

d.columns = ['subject', 'alpha', 'library size', 'median expression length', 'median compression rate']

import seaborn as sns

f,ax = plt.subplots(figsize=(4,3.5))
sns.pointplot(data=d, x='alpha', y='median compression rate', hue='subject', scale=0.5, palette='dark', ax=ax)
# TODO doesn't work; make it work
# for line in ax.lines[-1:]:  # Only adjust the last set of lines, which belong to the second plot
#     line.set_alpha(0.8)

sns.pointplot(data=d, x='alpha', y='median compression rate', color='k', scale=2, ax=ax)
plt.setp(ax.lines, zorder=100)
plt.setp(ax.collections, zorder=100, label="")

ax.set_xlabel(r'$\alpha$' + '\n(abstraction parameter)')
ax.invert_xaxis()

ax.set_ylabel('median\ncompression rate')

# ax.set_ylim(0,2500)
# ax.set_yticks([0,1000,2000])
ax.legend_.remove()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

f.savefig('XXX_compressionrate_persubject.png', dpi=500)
plt.close('all')

# ################################################################################
#

d = pd.DataFrame()

for subject in subjects:
    batch = data[subject]

    alphas = [1]
    size_library = []
    mean_expression_length = []
    for alpha in alphas:
        print(subject)
        print(alpha)
        compressor = ProbZip(alpha=alpha)
        compressor.compress_batch(batch)

        len_uncompressed = []
        len_compressed = []
        for x in range(len(batch)):
            encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(batch[x])
            len_uncompressed = len(batch[x])
            len_compressed = len(encoded_sequence)

            d = d.append(pd.Series([subject, len_uncompressed, len_compressed]), ignore_index=True)

d.columns=['subject', 'uncompressed bout length', 'compressed bout length']

g = sns.lmplot(data=d,height=4,legend=False,scatter_kws={'alpha':0.2,'linewidths':0}, x='uncompressed bout length', y='compressed bout length', hue='subject', palette='dark')

# Setting the limits for x and y axes
g.set(xlim=(0, 200), ylim=(0, 200))

g.set(xticks=[0,100,200], yticks=[0,100,200])

# Setting the labels for x and y axes
g.set_axis_labels('uncompressed\nbout length', 'compressed\nbout length')

# Iterating over each axis in the FacetGrid and adding the line
for ax in g.axes.flat:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, ls='--', c='k')

plt.tight_layout()
plt.savefig('XXX_compressed_vs_uncompressed_length.png', dpi=500)
plt.close('all')
