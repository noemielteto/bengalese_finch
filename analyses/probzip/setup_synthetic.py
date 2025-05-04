from concurrent.futures import ThreadPoolExecutor
from bengalese_finch.models.probzip import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gc

groundtruth_models_dir = 'bengalese_finch/analyses/probzip/groundtruth_models_synthetic/'

alpha = 100
pseudocounts = 10000

# Example 1: High-order dependency: p(c|a,b) = .5

terminals = ['<', '>', 'a', 'b', 'c']

compressor = ProbZip(alpha=alpha)
compressor.get_terminals(terminals)

compressor.library["['a', 'b']"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['b'], rate=None)
compressor.library["['<', ['a', 'b']]"] = Node(alpha=compressor.alpha, parent=compressor.library['<'], suffix=compressor.library["['a', 'b']"], rate=None)
compressor.library["[['<', ['a', 'b']], 'c']"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a', 'b']]"], suffix=compressor.library['c'], rate=None)
compressor.library["[['<', ['a', 'b']], '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a', 'b']]"], suffix=compressor.library['>'], rate=None)
compressor.library["[[['<', ['a', 'b']], 'c'], '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["[['<', ['a', 'b']], 'c']"], suffix=compressor.library['>'], rate=None)

compressor.library["['a', 'b']"].count = pseudocounts
compressor.library["['<', ['a', 'b']]"].count = pseudocounts
compressor.library["[['<', ['a', 'b']], 'c']"].count = pseudocounts
compressor.library["[['<', ['a', 'b']], '>']"].count = pseudocounts
compressor.library["[[['<', ['a', 'b']], 'c'], '>']"].count = pseudocounts

with open(f'{groundtruth_models_dir}model_0.pkl', 'wb') as f:
    pickle.dump(compressor, f)

# # Example 2: High-order dependency: p(c|a,b) = .5

# terminals = ['<', '>', 'a', 'b', 'c']

# compressor = ProbZip(alpha=alpha)
# compressor.get_terminals(terminals)

# compressor.library["['a', 'b']"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['b'], rate=None)
# compressor.library["['<', ['a', 'b']]"] = Node(alpha=compressor.alpha, parent=compressor.library['<'], suffix=compressor.library["['a', 'b']"], rate=None)
# compressor.library["[['<', ['a', 'b']], 'c']"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a', 'b']]"], suffix=compressor.library['c'], rate=None)
# compressor.library["[['<', ['a', 'b']], '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a', 'b']]"], suffix=compressor.library['>'], rate=None)
# compressor.library["[[['<', ['a', 'b']], 'c'], '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["[['<', ['a', 'b']], 'c']"], suffix=compressor.library['>'], rate=None)

# compressor.library["['a', 'b']"].count = pseudocounts
# compressor.library["['<', ['a', 'b']]"].count = pseudocounts
# compressor.library["[['<', ['a', 'b']], 'c']"].count = pseudocounts/2
# compressor.library["[['<', ['a', 'b']], '>']"].count = pseudocounts/2
# compressor.library["[[['<', ['a', 'b']], 'c'], '>']"].count = pseudocounts

# with open(f'{groundtruth_models_dir}model_1.pkl', 'wb') as f:
#     pickle.dump(compressor, f)

# Example 2: Repeats

terminals = ['<', '>', 'a', 'b']

compressor = ProbZip(alpha=alpha)
compressor.get_terminals(terminals)

compressor.library["['a']*4"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=None, rate=4)
compressor.library["['<', ['a']*4]"] = Node(alpha=compressor.alpha, parent=compressor.library['<'], suffix=compressor.library["['a']*4"], rate=None)
compressor.library["['b']*2"] = Node(alpha=compressor.alpha, parent=compressor.library['b'], suffix=None, rate=2)
compressor.library["[['b']*2, '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["['b']*2"], suffix=compressor.library['>'], rate=None)
compressor.library["[['<', ['a']*4], [['b']*2, '>']]"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a']*4]"], suffix=compressor.library["[['b']*2, '>']"], rate=None)

compressor.library["['a']*4"].count = pseudocounts
compressor.library["['<', ['a']*4]"].count = pseudocounts
compressor.library["['b']*2"].count = pseudocounts
compressor.library["[['b']*2, '>']"].count = pseudocounts
compressor.library["[['<', ['a']*4], [['b']*2, '>']]"].count = pseudocounts

with open(f'{groundtruth_models_dir}model_1.pkl', 'wb') as f:
    pickle.dump(compressor, f)

# Example 3: Repeats of chunks

terminals = ['<', '>', 'a', 'b', 'c']

compressor = ProbZip(alpha=alpha)
compressor.get_terminals(terminals)

compressor.library["['a']*4"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=None, rate=4)
compressor.library["['b', 'c']"] = Node(alpha=compressor.alpha, parent=compressor.library['b'], suffix=compressor.library['c'], rate=None)
compressor.library["['b', 'c']*2"] = Node(alpha=compressor.alpha, parent=compressor.library["['b', 'c']"], suffix=None, rate=2)
compressor.library["['<', ['a']*4]"] = Node(alpha=compressor.alpha, parent=compressor.library['<'], suffix=compressor.library["['a']*4"], rate=None)
compressor.library["[['b', 'c']*2, '>']"] = Node(alpha=compressor.alpha, parent=compressor.library["['b', 'c']*2"], suffix=compressor.library['>'], rate=None)
compressor.library["[['<', ['a']*4], [['b', 'c']*2, '>']]"] = Node(alpha=compressor.alpha, parent=compressor.library["['<', ['a']*4]"], suffix=compressor.library["[['b', 'c']*2, '>']"], rate=None)

compressor.library["['a']*4"].count = pseudocounts
compressor.library["['b', 'c']"].count = pseudocounts
compressor.library["['b', 'c']*2"].count = pseudocounts
compressor.library["['<', ['a']*4]"].count = pseudocounts
compressor.library["[['b', 'c']*2, '>']"].count = pseudocounts
compressor.library["[['<', ['a']*4], [['b', 'c']*2, '>']]"].count = pseudocounts

with open(f'{groundtruth_models_dir}model_2.pkl', 'wb') as f:
    pickle.dump(compressor, f)