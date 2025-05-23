from bengalese_finch.models.probzip import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = 'ababab'
dataset = [data] * 1000
compressor = ProbZip(alpha_vector=[1]*1000)
compressor.get_terminals(data)
symbol = compressor.compress(data, update_counts=False)
symbol.probability(data, 0)
results_dict = compressor.compress_dataset(dataset, steps=10000, log_every=1000)
symbol = compressor.compress(data, update_counts=False)
symbol
symbol.probability(data, 0)

for k, v in compressor.library.items():
    print(f'{k}: {v.count}')


data = 'abcd'

compressor = ProbZip(alpha_vector=[1]*1000)
compressor.get_terminals(data)
symbol = compressor.compress(data, update_counts=False)
symbol.probability(data, 0)

data = 'abcd'
dataset = [data] *1000
compressor = ProbZip(alpha_vector=[1]*1000)
compressor.get_terminals(data)
symbol = compressor.compress('abcd', update_counts=False)
symbol.probability(data, 0)

compressor.library["['a', 'b']"] = Node(alpha_vector=compressor.alpha_vector,
                                        parent=compressor.library['a'],
                                        suffix=compressor.library['b'],
                                        rate=None)
compressor.library["[['a', 'b'], 'c']"] = Node(alpha_vector=compressor.alpha_vector,
                                        parent=compressor.library["['a', 'b']"],
                                        suffix=compressor.library['c'],
                                        rate=None)
compressor.library["[[['a', 'b'], 'c'], 'd']"] = Node(alpha_vector=compressor.alpha_vector,
                                        parent=compressor.library["[['a', 'b'], 'c']"],
                                        suffix=compressor.library['d'],
                                        rate=None)

results_dict = compressor.compress_dataset(dataset, steps=10000, log_every=1000)
symbol = compressor.compress('abcd', update_counts=False)
symbol.probability(data, 0)

compressor.get_dataset_ll(dataset)

for k, v in compressor.library.items():
    print(f'{k}: {v.count}')

# compressor  = ProbZip(alpha=.000001)
# compressor.get_terminals(flatten_arbitrarily_nested_lists(dataset))
# compressor.library["['n']*2"] = Node(alpha=compressor.alpha, parent=compressor.library['n'], suffix=None, rate=2)
# compressor.library["['n']*8"] = Node(alpha=compressor.alpha, parent=compressor.library['n'], suffix=None, rate=8)
# compressor.library["['a', 'b']"] = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['b'], rate=None)
# compressor.library["[['a', 'b'], 'l']"] = Node(alpha=compressor.alpha, parent=compressor.library["['a', 'b']"], suffix=compressor.library['l'], rate=None)
# compressor.library["[[['a', 'b'], 'l'], 'e']"] = Node(alpha=compressor.alpha, parent=compressor.library["[['a', 'b'], 'l']"], suffix=compressor.library['e'], rate=None)
# compressor.library["['d', 'g']"] = Node(alpha=compressor.alpha, parent=compressor.library['d'], suffix=compressor.library['g'], rate=None)
# compressor.library["[['d', 'g'], 'f']"] = Node(alpha=compressor.alpha, parent=compressor.library["['d', 'g']"], suffix=compressor.library['f'], rate=None)
# compressor.library["[[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]"] = Node(alpha=compressor.alpha, parent=compressor.library["[[['a', 'b'], 'l'], 'e']"], suffix=compressor.library["[['d', 'g'], 'f']"], rate=None)
# compressor.library["[[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]*3"] = Node(alpha=compressor.alpha, parent=compressor.library["[[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]"], suffix=None, rate=3)
# compressor.library["[['n']*8, [[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]*3]"] = Node(alpha=compressor.alpha, parent=compressor.library["['n']*8"], suffix=compressor.library["[[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]*3"], rate=None)
# compressor.library["['<', [['n']*8, [[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]*3]]"] = Node(alpha=compressor.alpha, parent=compressor.library["<"], suffix=compressor.library["[['n']*8, [[[['a', 'b'], 'l'], 'e'], [['d', 'g'], 'f']]*3]"], rate=None)

# compressor.compress(dataset_test[10])

ll_test = compressor.get_dataset_ll(dataset_test)
print(ll_test)
# compressor.library


print('compression done')
ll_train = compressor.get_dataset_ll(dataset_train)
ll_test = compressor.get_dataset_ll(dataset_test)
print('computed ll')
entropy_train = compressor.get_dataset_shannon_entropy(dataset_train)
entropy_test = compressor.get_dataset_shannon_entropy(dataset_test)
print('computed entropy')


print(len(compressor.library))
print(len(compressor.get_important_library()))
print(compressor.library['a'].children)
print([c.count for c in compressor.library['a'].children])
print(compressor.library['a'].children[0])
print(compressor.library['b'].children[0])
print(compressor.library['l'].children[0])
print(compressor.library['e'].children[0])
print(compressor.library['d'].children[0])
print(compressor.library['g'].children[0])
print(compressor.library['f'].children[0])
print(compressor.library['n'].children[0])

############

triplets     = ['abc', 'def', 'ghi', 'jkl', 'mno']
dataset       = [''.join(np.random.choice(triplets, 50)) for _ in range(100)]
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=42)

compressor = ProbZip(alpha=[1]*1000)
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

results_dict = compressor.search_add_remove(dataset_train=dataset_train,
                                            dataset_val=dataset_val,
                                            steps=10,
                                            log_every=1)

############

# compressor  = ProbZip(alpha=1)
# compressor.get_terminals(['a','b','c','d'])
# node = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['b'])
# compressor.library[node.expression] = node
# node = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['c'])
# compressor.library[node.expression] = node
# node = Node(alpha=compressor.alpha, parent=compressor.library['a'], suffix=compressor.library['d'])
# compressor.library[node.expression] = node

# node = Node(alpha=compressor.alpha, parent=compressor.library["['a', 'b']"], suffix=compressor.library["['a', 'c']"])
# compressor.library[node.expression] = node
# node = Node(alpha=compressor.alpha, parent=compressor.library["['a', 'b']"], suffix=compressor.library["['a', 'd']"])
# compressor.library[node.expression] = node

# compressor.library['a'].count = 1
# compressor.library['b'].count = 1
# compressor.library['c'].count = 1
# compressor.library['d'].count = 1
# compressor.library["['a', 'b']"].count = 10
# compressor.library["['a', 'c']"].count = 1
# compressor.library["['a', 'd']"].count = 1

# compressor.library["[['a', 'b'], ['a', 'c']]"].count = 1
# compressor.library["[['a', 'b'], ['a', 'd']]"].count = 1

# data = 'abac'
# i = 0

# print(compressor.library[''].probability(data, i))
# print(compressor.library["['a', 'b']"].probability(data, i))

# l = [compressor.library[''].probability(data, i),
#      compressor.library['a'].probability(data, i),
#      compressor.library['b'].probability(data, i),
#      compressor.library['c'].probability(data, i),
#      compressor.library['d'].probability(data, i),
#      compressor.library["['a', 'b']"].probability(data, i),
#      compressor.library["['a', 'c']"].probability(data, i),
#      compressor.library["['a', 'd']"].probability(data, i),
#      compressor.library["[['a', 'b'], ['a', 'c']]"].probability(data, i),
#      compressor.library["[['a', 'b'], ['a', 'd']]"].probability(data, i)
#     ]

# print(l)
# print(sum(l))

# ##############################################################################

# compressor  = ProbZip(alpha=1)
# synthetic_bout = 'Ynnnnnnnabledgfabledgfnnfabledgf'
# dataset       = [synthetic_bout] * 5000
# compressor.compress_dataset(dataset)
# important_library = compressor.get_important_library()
# print('---------------------')
# print(important_library)

# # ##############################################################################

# # # learn repeat phrase data
# compressor  = ProbZip(alpha=100)
# repeats     = ['a', 'bb', 'ccc', 'dddd', 'eeeee', 'ffffff', 'ggggggg', 'hhhhhhhh', 'iiiiiiiii', 'jjjjjjjjjj']
# data        = '' + ''.join(np.random.choice(repeats, 1000))
# dataset       = [data]
# compressor.compress_dataset(dataset)
# important_library = compressor.get_important_library(compressor.library, threshold=10)
# print('---------------------')
# print(important_library)

# ##############################################################################

# # learn bag of duplets data
# compressor  = ProbZip(alpha=10)
# duplets     = ['ab', 'cd', 'ef', 'gh', 'ij']
# data        = ''.join(np.random.choice(duplets, 1000))
# dataset       = [data]
# compressor.compress_dataset(dataset)

# print(compressor)
# # list(compressor.library["['a', 'b']"].probability_distribution_of_parents(data, 0).values()) + [compressor.library["['a', 'b']"].probability(data, 0)]
# # compressor.library['a'].probability_distribution_of_parents(data, 0)
# # encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(data, verbose=True)
# #
# important_library = get_important_library(compressor.library, threshold=10)
# print('---------------------')
# print(important_library)
# # # i=0
# # # for x in range(10):
# # #     print(compressor.library['a'].update(data, i))
#
# # encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(data, verbose=True)
# # print(encoded_sequence)
#
# # ############################################################################
#
# # # learn bag of triplets data
# # compressor  = ProbZip(alpha=100)
# # triplets     = ['abc', 'def', 'ghi', 'jkl', 'mno']
# # data        = 'abc' + ''.join(np.random.choice(triplets, 1000))
# # dataset       = [data]
# # compressor.compress_dataset(dataset)
#
# # print(compressor)
#
# # # i=0
# # # print(compressor.library['a'].update('abc', i))
#
# # encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(data, verbose=True)
# # print(encoded_sequence)
#
# # ############################################################################
#
# # learn repeat pf chunks data
# # compressor  = ProbZip(alpha=10)
# # data        = 'abcd'*10
# # dataset       = [data]*100
# # compressor.compress_dataset(dataset)
#
# # print(compressor)
#
# # # i=0
# # # print(compressor.library['a'].update('abc', i))
#
# # encoded_sequence, encoded_sequence_flat = compressor.encode_sequence(data, verbose=True)
# # print(encoded_sequence)
# # print(encoded_sequence_flat)
#
# # ############################################################################
#
# compressor  = ProbZip(alpha=1)

# dataset = ['<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfnnfabledgf>',
#         '<nnnnabledgfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
#         '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
#         '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
#         '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>'
#         ]

# compressor.compress_dataset(dataset)

# likelihoods = []
# for alpha in [0.001, 0.1, 1, 10, 100]:
#     print(f'alpha: {alpha}')
#     compressor = ProbZip(alpha=alpha)
#     compressor.compress_dataset(dataset)
#     likelihood = compressor.get_dataset_likelihood(dataset)
#     likelihoods.append(likelihood)

# plt.plot(likelihoods)
# plt.show()
