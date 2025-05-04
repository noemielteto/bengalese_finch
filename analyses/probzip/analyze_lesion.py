from bengalese_finch.models.probzip import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon

colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40', '#6D545A', '#403BAF', '#8B8502', '#46AB89']

#################################################################################

models_dir = 'bengalese_finch/analyses/probzip/fitted_models/'
figs_dir = 'bengalese_finch/analyses/probzip/figs/lesion_lenmatched/'
data = get_data_lesion(strings=True)
subjects = list(data.keys())

# Plot the mean and the actual data points, as well as a t-test for pre-post song length difference
f, ax = plt.subplots(1, len(subjects), figsize=(3*len(subjects), 3), sharex=True, sharey=True)
song_len_distribution = [
    len(x) for subject in subjects for x in data[subject]['prelesion'][:100] + data[subject]['postlesion'][:100]
]
line_pos = max(song_len_distribution) * 1.01
p_pos = line_pos * 1.05
for i, subject in enumerate(subjects):
    data_pre = data[subject]['prelesion'][:100]
    data_post = data[subject]['postlesion'][:100]
    song_len_distribution_pre = [len(x) for x in data_pre]
    song_len_distribution_post = [len(x) for x in data_post]
    # Plot the mean song length
    ax[i].bar([0, 1],
            [np.mean(song_len_distribution_pre), np.mean(song_len_distribution_post)],
            tick_label=['pre', 'post'],
            color=['grey', 'grey']
            )
    # Add songs as data points
    ax[i].scatter([0] * len(song_len_distribution_pre), song_len_distribution_pre, color='k', alpha=0.1)
    ax[i].scatter([1] * len(song_len_distribution_post), song_len_distribution_post, color='k', alpha=0.1)
    # Run significance test
    _, p = wilcoxon(song_len_distribution_pre, song_len_distribution_post)
    # Add p-value itself
    if p < 0.001:
        ax[i].text(0.5, p_pos, f'p < .001',
            ha='center', va='bottom', fontsize=16, color='black')
    else:
        ax[i].text(0.5, p_pos, f'p = {p:.3f}',
            ha='center', va='bottom', fontsize=16, color='black')
    ax[i].plot([0, 1], [line_pos]*2, color='black', lw=2)
    ax[i].set_title(subject, fontsize=16)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
ax[0].set_ylabel('song length\n(#syllables)')
ax[0].set_ylim(0, max(song_len_distribution) * 1.5)
plt.tight_layout()
plt.savefig(f'{figs_dir}lesion_song_length_per_subject.png', dpi=300)
plt.savefig(f'{figs_dir}lesion_song_length_per_subject.pdf', dpi=300)
plt.close('all')


f, ax = plt.subplots(1, len(subjects), figsize=(3*len(subjects), 3), sharex=True, sharey=True)
for i, subject in enumerate(subjects):
    data_pre = data[subject]['prelesion']
    data_post = data[subject]['postlesion']
    song_lengths_pre = [len(x) for x in data_pre]
    song_lengths_post = [len(x) for x in data_post]
    # Plot individual song lengths
    if i==len(subjects)-1:
        ax[i].scatter(np.arange(len(song_lengths_pre)), song_lengths_pre, color='green', alpha=0.1, label='pre-lesion')
        ax[i].scatter(np.arange(len(song_lengths_post)), song_lengths_post, color='red', alpha=0.1, label='post-lesion')
    else:
        ax[i].scatter(np.arange(len(song_lengths_pre)), song_lengths_pre, color='green', alpha=0.1)
        ax[i].scatter(np.arange(len(song_lengths_post)), song_lengths_post, color='red', alpha=0.1)
    # Plot rolling mean
    rolling_mean_pre = pd.Series(song_lengths_pre).rolling(window=20).mean()
    rolling_mean_post = pd.Series(song_lengths_post).rolling(window=20).mean()
    ax[i].plot(rolling_mean_pre, color='green', lw=3)
    ax[i].plot(rolling_mean_post, color='red', lw=3)

    ax[i].set_title(subject, fontsize=16)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].axvline(x=100, color='black', lw=1)
    ax[i].set_xlabel('bouts')
ax[0].set_ylabel('song length\n(#syllables)')
ax[-1].legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig(f'{figs_dir}lesion_song_length_by_time_per_subject.png', dpi=300)
plt.savefig(f'{figs_dir}lesion_song_length_by_time_per_subject.pdf', dpi=300)
plt.close('all')

def get_models_results(models_dir):
    # Construct the models and results dictionaries
    models = {}
    models_results = {}
    for subject in subjects:
        models[subject] = {'prelesion': [], 'postlesion': []}
        models_results[subject] = {'prelesion': [], 'postlesion': []}
        for phase in ['prelesion', 'postlesion']:
            for model_i in range(10):
                try:
                    with open(f'{models_dir}model_{subject}_{phase}_{model_i}_lenmatched.pkl', 'rb') as f:
                        compressor = pickle.load(f)
                except FileNotFoundError:
                    print(f'Model file not found for subject {subject}, phase {phase}, model {model_i}. Skipping.')
                    continue
                try:
                    with open(f'{models_dir}results_{subject}_{phase}_{model_i}_lenmatched.pkl', 'rb') as f:
                        results_dict = pickle.load(f)
                except FileNotFoundError:
                    print(f'Results file not found for subject {subject}, phase {phase}, model {model_i}. Skipping.')
                    continue

                models[subject][phase].append(compressor)
                models_results[subject][phase].append(results_dict)

    return models, models_results

def plot(models, models_results, library_threshold=1):
    """
    Plot the results of the models.
    """
    subjects = list(models_results.keys())
    i_model_pre = [get_best_model_i(models_results[subject]['prelesion']) for subject in subjects]
    i_model_post = [get_best_model_i(models_results[subject]['postlesion']) for subject in subjects]
    mdl_data_pre = [models_results[subject]['prelesion'][i_model_pre[i]]['mdl_data_test'][-1] for i, subject in enumerate(subjects)]
    mdl_data_post = [models_results[subject]['postlesion'][i_model_post[i]]['mdl_data_test'][-1] for i, subject in enumerate(subjects)]
    mdl_library_pre = [models_results[subject]['prelesion'][i_model_pre[i]]['mdl_library'][-1] for i, subject in enumerate(subjects)]
    mdl_library_post = [models_results[subject]['postlesion'][i_model_post[i]]['mdl_library'][-1] for i, subject in enumerate(subjects)]
    # mdl_pre = [models_results[subject]['prelesion'][i_model_pre[i]]['mdl'][-1] for i, subject in enumerate(subjects)]
    # mdl_post = [models_results[subject]['postlesion'][i_model_post[i]]['mdl'][-1] for i, subject in enumerate(subjects)]
    library_size_pre = [len(models[subject]['prelesion'][i_model_pre[i]].get_important_library(threshold=library_threshold)) for i, subject in enumerate(subjects)]
    library_size_post = [len(models[subject]['postlesion'][i_model_post[i]].get_important_library(threshold=library_threshold)) for i, subject in enumerate(subjects)]

    library_hierarchy_distribution_pre = [get_library_hierarchy_distribution(models[subject]['prelesion'][i_model_pre[i]], library_threshold) for i, subject in enumerate(subjects)]
    library_hierarchy_distribution_post = [get_library_hierarchy_distribution(models[subject]['postlesion'][i_model_post[i]], library_threshold) for i, subject in enumerate(subjects)]

    library_hierarchy_pre = [np.max(x) for x in library_hierarchy_distribution_pre]
    library_hierarchy_post = [np.max(x) for x in library_hierarchy_distribution_post]

    library_order_distribution_pre = [get_library_order_distribution(models[subject]['prelesion'][i_model_pre[i]], library_threshold) for i, subject in enumerate(subjects)]
    library_order_distribution_post = [get_library_order_distribution(models[subject]['postlesion'][i_model_post[i]], library_threshold) for i, subject in enumerate(subjects)]

    library_order_pre = [np.max(x) for x in library_order_distribution_pre]
    library_order_post = [np.max(x) for x in library_order_distribution_post]

    data_to_plot = {(0, 0): [mdl_data_pre, mdl_data_post],
                    (0, 1): [mdl_library_pre, mdl_library_post],
                    (1, 0): [library_hierarchy_pre, library_hierarchy_post],
                    (1, 1): [library_order_pre, library_order_post]}

    _, ax = plt.subplots(2, 2, figsize=(6, 6))

    for (i, j), (data_pre, data_post) in data_to_plot.items():

        ax[i][j].bar([0, 1],
                [np.mean(data_pre), np.mean(data_post)],
                tick_label=['pre', 'post'],
                color=['grey', 'grey']
                )
        # Add subjects as data points
        for i_subject, _ in enumerate(subjects):
            xshift = .05 * i_subject
            xpos = [-.15 + xshift, .85 + xshift]
            ax[i][j].scatter(xpos, [data_pre[i_subject], data_post[i_subject]], color=colors[i_subject], alpha=1, label=subjects[i_subject])
            ax[i][j].plot(xpos, [data_pre[i_subject], data_post[i_subject]], color='black', alpha=1, lw=0.5)

        # Run significance test
        _, p = wilcoxon(data_pre, data_post)
        # Add p-value itself
        ax[i][j].text(0.5, max(np.mean(data_pre), np.mean(data_post)) * 2, f'p = {p:.3f}',
                ha='center', va='bottom', fontsize=16, color='black') 

        # Add horizontal line between the two bars
        ax[i][j].plot([0, 1], [max(np.mean(data_pre), np.mean(data_post)) * 1.8] * 2, color='black', lw=2)
        print(f'Wilcoxon test for {i}: p = {p}')

        ax[i][j].set_xticks([0, 1])
        ax[i][j].spines['top'].set_visible(False)
        ax[i][j].spines['right'].set_visible(False)

    ax[1][0].set_xticklabels(['pre', 'post'])
    ax[1][1].set_xticklabels(['pre', 'post'])

    ax[0][0].set_yticks([0, 4])
    ax[0][1].set_yticks([0, 0.015])
    ax[1][0].set_yticks([0, 10])
    ax[1][1].set_yticks([0, 10])

    ax[0][0].set_ylabel('bits/syllable')
    ax[0][0].set_title('test data\n')
    ax[0][1].set_title('ProbZip library\n')
    ax[1][0].set_ylabel('library hierarchy')
    ax[1][1].set_ylabel('library order')
    # Legend on the right
    ax[1][1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=16)
    plt.tight_layout()
    # Save both in png and pdf formats
    plt.savefig(f'{figs_dir}lesion_result.png', dpi=300)
    plt.savefig(f'{figs_dir}lesion_result.pdf', dpi=300)
    plt.close('all')

# TODO(noemielteto): Goes in analysis library
def get_best_model_i(results):
    best_model_i = np.argmin([r['mdl'][-1] for r in results])
    return best_model_i

def get_library_order_distribution(compressor, library_threshold=1):
    important_library = compressor.get_important_library(threshold=library_threshold)
    library_order_distribution = []
    for symbol in important_library.values():
        library_order_distribution.append(symbol.order)
    return library_order_distribution

def get_library_hierarchy_distribution(compressor, library_threshold=1):
    important_library = compressor.get_important_library(threshold=library_threshold)
    library_hierarchy_distribution = [node.get_hierarchy() for node in important_library.values()]
    return library_hierarchy_distribution

def get_mdl_per_hierarchy(compressor, dataset_test, library_threshold=1):
    test_n = len(flatten_arbitrarily_nested_lists(dataset_test))
    important_library = compressor.get_important_library(threshold=library_threshold)
    mdl_per_hierarchy = {}
    for node in important_library.values():
        h = node.get_hierarchy()
        if h not in mdl_per_hierarchy:
            mdl_per_hierarchy[h] = 0
        print(h)
        print(node.get_entropy())
        mdl_per_hierarchy[h] += node.get_entropy()/test_n
    return mdl_per_hierarchy

models, models_results = get_models_results(models_dir)

entropy_per_hierarchy_pre_all = {}
entropy_per_hierarchy_post_all = {}
for subject in subjects:
    best_model_i_pre = get_best_model_i(models_results[subject]['prelesion'])
    best_model_i_post = get_best_model_i(models_results[subject]['postlesion'])

    model_pre = models[subject]['prelesion'][best_model_i_pre]
    model_post = models[subject]['postlesion'][best_model_i_post]

    subject_dataset_pre = data[subject]['prelesion']
    subject_dataset_post = data[subject]['postlesion']
    dataset_train_pre, dataset_test_pre = train_test_split(subject_dataset_pre,
                                                test_size=0.2,
                                                random_state=best_model_i_pre+10)
    dataset_val_pre, dataset_test_pre = train_test_split(dataset_test_pre,
                                                test_size=0.5,
                                                random_state=best_model_i_pre+10)
    dataset_train_post, dataset_test_post = train_test_split(subject_dataset_post,
                                                test_size=0.2,
                                                random_state=best_model_i_post+10)
    dataset_val_post, dataset_test_post = train_test_split(dataset_test_post,
                                                test_size=0.5,
                                                random_state=best_model_i_post+10)
    
    # get entropy per hierarchy
    entropy_per_hierarchy_pre = get_mdl_per_hierarchy(model_pre, dataset_test_pre, library_threshold=1)
    entropy_per_hierarchy_post = get_mdl_per_hierarchy(model_post, dataset_test_post, library_threshold=1)

    # add to all
    entropy_per_hierarchy_pre_all[subject] = entropy_per_hierarchy_pre
    entropy_per_hierarchy_post_all[subject] = entropy_per_hierarchy_post

n_hierarchies = len(entropy_per_hierarchy_pre_all)

# Plot the entropy per hierarchy
f, ax = plt.subplots(n_hierarchies, 1, figsize=(4, 8), sharex=True, sharey=True)
for i, hierarchy in enumerate(sorted(entropy_per_hierarchy_pre_all[subjects[0]].keys())):
    
    # Plot the mean across subjects as barplot
    ax[i].bar([0, 1],
              [np.mean([entropy_per_hierarchy_pre_all[subject][hierarchy] 
                        for subject in subjects if hierarchy in entropy_per_hierarchy_pre_all[subject]]),
               np.mean([entropy_per_hierarchy_post_all[subject][hierarchy] 
                        for subject in subjects if hierarchy in entropy_per_hierarchy_post_all[subject]])],
              tick_label=['pre', 'post'],
              color=['grey', 'grey']
              )

    # Add subjects as data points
    for j, subject in enumerate(subjects):
        if hierarchy in entropy_per_hierarchy_pre_all[subject]:
            ax[i].scatter([0], entropy_per_hierarchy_pre_all[subject][hierarchy],
                        color=colors[j], alpha=0.5)
        if hierarchy in entropy_per_hierarchy_post_all[subject]:
            ax[i].scatter([1], entropy_per_hierarchy_post_all[subject][hierarchy],
                        color=colors[j], alpha=0.5)
        if hierarchy in entropy_per_hierarchy_pre_all[subject] and hierarchy in entropy_per_hierarchy_post_all[subject]:
            # Add line connecting pre and post
            ax[i].plot([0, 1],
                   [entropy_per_hierarchy_pre_all[subject][hierarchy], entropy_per_hierarchy_post_all[subject][hierarchy]],
                    color='k', alpha=1, lw=1)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
ax[i].set_ylabel('bits/syllable')
# Log scale
ax[i].set_yscale('log')
plt.tight_layout()
plt.savefig(f'{figs_dir}lesion_mld_per_hierarchy.png', dpi=300)
plt.savefig(f'{figs_dir}lesion_mdl_per_hierarchy.pdf', dpi=300)
plt.close('all')

# Different plot where mean entropy per hierarchy is plotted as heatmap; two columns are pre and post; each row is a hierarchy level
f, ax = plt.subplots(1, 1, figsize=(4, 4))
entropy_per_hierarchy = np.zeros((n_hierarchies, 2))
for i, hierarchy in enumerate(sorted(entropy_per_hierarchy_pre_all[subjects[0]].keys())):
    entropy_per_hierarchy[i][0] = np.mean([entropy_per_hierarchy_pre_all[subject][hierarchy] 
                        for subject in subjects if hierarchy in entropy_per_hierarchy_pre_all[subject]])
    entropy_per_hierarchy[i][1] = np.mean([entropy_per_hierarchy_post_all[subject][hierarchy]
                        for subject in subjects if hierarchy in entropy_per_hierarchy_post_all[subject]])
# Plot as heatmap
sns.heatmap(entropy_per_hierarchy, cmap='Greys', cbar=False, ax=ax)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['pre', 'post'])
ax.set_yticks(np.arange(n_hierarchies)+.5)
# ax.set_yticklabels(sorted(entropy_per_hierarchy_pre_all[subjects[0]].keys()))
ax.set_ylabel('hierarchy')
# reverse y-axis
ax.invert_yaxis()
# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('log bits/syllable', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(f'{figs_dir}lesion_mdl_per_hierarchy_heatmap.png', dpi=300)
plt.savefig(f'{figs_dir}lesion_mdl_per_hierarchy_heatmap.pdf', dpi=300)
plt.close('all')

from matplotlib.colors import LinearSegmentedColormap

# Hierarchy plotted as heatmap for each subject
f, ax = plt.subplots(1, len(subjects), figsize=(4*len(subjects), 4), sharex=True, sharey=True)
for i, subject in enumerate(subjects):
    entropy_per_hierarchy = np.zeros((n_hierarchies, 2))
    for j, hierarchy in enumerate(sorted(entropy_per_hierarchy_pre_all[subject].keys())):
        if hierarchy in entropy_per_hierarchy_pre_all[subject]:
            entropy_per_hierarchy[j][0] = entropy_per_hierarchy_pre_all[subject][hierarchy]
        if hierarchy in entropy_per_hierarchy_post_all[subject]:
            entropy_per_hierarchy[j][1] = entropy_per_hierarchy_post_all[subject][hierarchy]
    
    # # Create a custom colormap based on the subject's color
    subject_color = colors[i]
    custom_cmap = LinearSegmentedColormap.from_list(f"subject_cmap_{i}", ['white', subject_color])
    # custom_cmap = 'Greys'
    
    # Plot as heatmap
    sns.heatmap(entropy_per_hierarchy, cmap=custom_cmap, cbar=False, ax=ax[i])
    ax[i].set_xticks([0.5, 1.5])
    ax[i].set_xticklabels(['pre', 'post'])
    ax[i].set_ylabel('hierarchy')
    # Reverse y-axis
    ax[i].invert_yaxis()
    ax[i].set_title(subject)
    # Add colorbar
    cbar = plt.colorbar(ax[i].collections[0], ax=ax[i])
    cbar.set_label('log bits/syllable', rotation=270, labelpad=15)
ax[0].set_yticks(np.arange(n_hierarchies)+.5)
# ax[0].set_yticklabels(sorted(entropy_per_hierarchy_pre_all[subjects[0]].keys()))
plt.tight_layout()
plt.savefig(f'{figs_dir}lesion_mdl_per_hierarchy_heatmap_per_subject.png', dpi=300)
plt.savefig(f'{figs_dir}lesion_mdl_per_hierarchy_heatmap_per_subject.pdf', dpi=300)
plt.close('all')

# Pre post change across all subjects

plot(models=models, models_results=models_results, library_threshold=.95)

# Pre post change in libraries of all subjects

for subject in subjects:
    model_pre = models[subject]['prelesion'][get_best_model_i(models_results[subject]['prelesion'])]
    model_post = models[subject]['postlesion'][get_best_model_i(models_results[subject]['postlesion'])]

    # Save libraries in plot
    model_pre.plot(save_name=f'{figs_dir}lesion_library_{subject}_pre.png')
    model_post.plot(save_name=f'{figs_dir}lesion_library_{subject}_post.png')

    # Save libraries in txt
    model_pre.write_to_txt(f'{figs_dir}lesion_library_{subject}_pre.txt')
    model_post.write_to_txt(f'{figs_dir}lesion_library_{subject}_post.txt')