import os, platform
from scipy.stats import pearsonr
import scipy
import pandas as pd
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt

if 'Windows' in platform.platform():
    slash = '\\'
else:
    slash = '/'

# Get the absolute directory of the current file (utils.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is one directory up from 'models'
project_root = os.path.dirname(current_dir)

def flatten(t):
    return [item for sublist in t for item in sublist]

def corrfunc(x,y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot - like seaborn pairplot."""
    """You will need to call g.map_lower(corrfunc) """
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    if p<.001:
        ax.annotate('r=' + str(np.round(r,2)) + '; p<.001', xy=(.1, .9), weight='bold', bbox=dict(facecolor='red', alpha=0.5), xycoords=ax.transAxes)
    elif p<.05:
        ax.annotate('r=' + str(np.round(r,2)) + '; p=' + str(np.round(p,3)), xy=(.1, .9), weight='bold', bbox=dict(facecolor='red', alpha=0.5), xycoords=ax.transAxes)
    else:
        ax.annotate('r=' + str(np.round(r,2)) + '; p=' + str(np.round(p,3)), xy=(.1, .9), xycoords=ax.transAxes)

def min_max_scale(a, lim_left=0, lim_right=1):
    scaled = lim_left + ((lim_right - lim_left) / (a.max() - a.min())) * (a-a.min())
    if a.min() < 0:
        return scaled - scaled.min() # special case if distribution has negative values
    else:
        return scaled

def shannon_entropy(events):
    return scipy.stats.entropy(pd.Series(events).value_counts())

def softmax(a):
    return np.exp(a)/sum(np.exp(a))

def my_normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def product_of_list(lst):
    return reduce(lambda x, y: x * y, lst)

def flatten_arbitrarily_nested_lists(nested_list):
    result = ""
    for element in nested_list:
        if isinstance(element, list):
            # If the element is a list, recursively flatten it
            result += flatten_arbitrarily_nested_lists(element)
        else:
            # If the element is not a list, concatenate it directly
            result += element
    return result

def rank_array(arr):
    # Argsort the array: indices of sorted elements
    sorted_indices = np.argsort(arr)
    # Create an empty array of the same shape to hold the ranks
    ranks = np.empty_like(sorted_indices, dtype=float)
    # Assign ranks, handling ties by assigning the average rank
    rank = 1
    while rank <= len(arr):
        # Find indices of elements that have the same value (ties)
        tied_indices = [sorted_indices[rank-1]]
        while rank < len(arr) and arr[sorted_indices[rank-1]] == arr[sorted_indices[rank]]:
            tied_indices.append(sorted_indices[rank])
            rank += 1
        # Calculate the average rank for tied elements
        average_rank = np.mean(range(rank, rank + len(tied_indices)))
        # Assign the average rank to all tied elements
        for idx in tied_indices:
            ranks[idx] = average_rank
        rank += len(tied_indices)
    return ranks

def get_elements(data, proportion_threshold=0.01, exclude_start_stop_tokens=True):
    elements, counts = np.unique(data, return_counts=True)
    proportions = counts/np.sum(counts)
    elements = list(elements[proportions>proportion_threshold])
    if exclude_start_stop_tokens:
        return [e for e in elements if e not in ['START','STOP']]
    else:
        return elements

def get_marginals(data, elements):
    counts = np.array([sum(np.array(data)==e) for e in elements])
    return counts/np.sum(counts)

def get_transition_matrix(data, elements):
    n = len(elements)

    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(data,data[1:]):
        if i in elements and j in elements:
            from_, to_ = elements.index(i), elements.index(j)
            M[from_][to_] += 1

    # convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def get_chunks_from_transition_matrix(M, elements, threshold=0.8):
    chunked_elements = []
    chunks = []
    indices = np.where(np.array(M)>threshold)
    for i,j in zip(list(indices[0]),list(indices[1])):
        print(elements[i])
        print(elements[j])

        if (elements[i] in ['START','STOP']) or (elements[j] in ['START','STOP']):
            continue

        if i in chunked_elements:
            for chunk in chunks:
                if elements[i] in chunk:
                    chunks.remove(chunk)
                    left = chunk
        else:
            left = elements[i]
            chunked_elements.append(i)

        if j in chunked_elements:
            for chunk in chunks:
                if elements[j] in chunk:
                    chunks.remove(chunk)
                    right = chunk
        else:
            right = elements[j]
            chunked_elements.append(j)

        chunks.append(left+right)

        print(chunks)
        print('---------')

    return chunks


def get_data_lesion(strings=False):

    data_dir = os.path.join(project_root, "data", f"lesion_data")

    data = {}
    subjects = ['bird1', 'bird2', 'bird3', 'bird4', 'bird5', 'bird6', 'bird7']

    for subject in subjects:
        data_label = subject
        data[data_label] = {}

        for phase in ['prelesion', 'postlesion']:

            data[data_label][phase] = []

            file_path = os.path.join(data_dir, f"{subject}_{phase}.txt")
            with open(file_path) as f:
                contents = f.read()
            if strings:
                for bout in contents.split('Y')[1:-1]:
                    data[data_label][phase].append('<' + bout + '>')
            else:
                for bout in contents.split('Y')[1:-1]:
                    data[data_label][phase].append(['START'] + list(bout) + ['STOP'])
    
    return data


def get_data(experimenter='Lena', phase='baseline', strings=False):

    data_dir = os.path.join(project_root, "data", f"{phase}_data")

    data = {}
    if experimenter=='Lena':
        subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']
    if experimenter=='Simon':
        subjects = ['rd6030', 'rd8031', 'rd6bu6', 'ye0wh0', 'rd5374']

    if phase=='baseline':

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            file_path = os.path.join(data_dir, f"{subject}.txt")
            with open(file_path) as f:
                contents = f.read()
            # we skip first and last separator because they produce empty strings

            if experimenter=='Lena':
                if strings:
                    for bout in contents.split('Y')[1:-1]:
                        if len(bout):
                            # data[data_label].append('<' + bout + '>')
                            # Note: Removed the <> brackets because they are not necessary for ProbZip! Reconsider if needed.
                            data[data_label].append(bout)
                else:
                    for bout in contents.split('Y')[1:-1]:
                        if len(bout):
                            data[data_label].append(['START'] + list(bout) + ['STOP'])
            if experimenter=='Simon':
                if strings:
                    for bout in contents.split('Y')[1:-1]:
                        if len(bout):
                            data[data_label].append('<' + bout + '>')
                else:
                    for bout in contents.split(','):
                        if len(bout):
                            data[data_label].append(['START'] + list(bout) + ['STOP'])

    elif phase=='post-training':

        for subject in subjects:
            for target in ['T1','T2']:
                data_label = f'{subject}_{target}'
                data[data_label] = []
                file_path = os.path.join(data_dir, f"{subject}.txt")
                with open(file_path) as f:
                    contents = f.read()
                # we skip first and last separator because they produce empty strings

                if experimenter=='Lena':
                    for bout in contents.split('Y')[1:-1]:
                        data[data_label].append(['START'] + list(bout) + ['STOP'])
                if experimenter=='Simon':
                    # To be implemented if we get the post-training raw data from him too
                    return

    elif phase=='synthetic':

        df = pd.read_csv(f'{data_dir}.csv')

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            subdf = df[df['subject']==subject]

            for bout_index in subdf.bout_index.unique():
                bout = subdf[subdf['bout_index']==bout_index].syllable.values
                data[data_label].append(['START'] + list(bout) + ['STOP'])

    return data

def get_timestamped_data(phase='baseline'):

    data = {}
    onsets = {}
    durations = {}
    latencies = {}
    subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']

    if phase=='baseline':

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            onsets[data_label] = []
            durations[data_label] = []
            latencies[data_label] = []

            data_dir = os.path.join(project_root, "data", f"timestamped_data")

            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]  # get list of CSV files
            for f in csv_files:
                df = pd.read_csv(os.path.join(data_dir, f))
                bout = [x.split('.')[0] for x in df.columns]

                data[data_label].append(['START'] + bout + ['STOP'])
                start = df.iloc[0].values
                end = df.iloc[1].values

                # set everything relative to the first onset (because sometimes there is a long quiet latency before the first syllable in the bout)
                end   = end - start[0]
                start = start - start[0]

                onsets[data_label].append([np.nan]            + list(start)             + [np.nan])
                durations[data_label].append([np.nan]         + list(end-start)             + [np.nan])
                latencies[data_label].append([np.nan, np.nan] + list(start[1:] - end[:-1])  + [np.nan])

    elif phase=='post-training':
        # TODO implement
        return True

    return data, onsets, durations, latencies


def get_joint_timestamped_data(include_durations=True, include_latencies=True):

    # this will discretize the durations and latencies into 10ms bins from 0 to 200ms;
    # TODO resolution could be a free parameter
    bins = np.linspace(0, 200, 41)
    len_bins = len(bins)

    all_data, all_onsets, all_durations, all_latencies = get_timestamped_data(phase='baseline')

    joint_timestamped_data = {}
    for subject in all_data.keys():

        joint_timestamped_data[subject] = []

        for i_bout in range(len(all_data[subject])):

            syllables = all_data[subject][i_bout]
            durations = all_durations[subject][i_bout]
            latencies = all_latencies[subject][i_bout]

            durations_bins = np.digitize(durations, bins)
            durations_digitized = [int(bins[i-1]) if i<len(bins) else np.nan for i in durations_bins]

            latencies_bins = np.digitize(latencies, bins)
            latencies_digitized = [int(bins[i-1]) if i<len(bins) else np.nan for i in latencies_bins]

            if not include_durations and not include_latencies:
                x = [str((syllables[i_syllable])) for i_syllable in range(len(syllables))]

            elif include_durations and not include_latencies:
                x = [str((syllables[i_syllable], durations_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            elif include_latencies and not include_durations:
                x = [str((syllables[i_syllable], latencies_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            else:
                x = [str((syllables[i_syllable], durations_digitized[i_syllable], latencies_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            joint_timestamped_data[subject].append(x)

    return joint_timestamped_data


def get_bout_borders(bouts):
    bout_borders = []
    start = 0
    for bout in bouts:
        bout_borders.append((start, start+len(bout)))
        start+=len(bout)
    return bout_borders

