from probzip.probzip import *

##############################################################################
#
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
compressor  = ProbZip(alpha=1)

dataset = ['<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
        '<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
        '<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfnnfabledgf>',
        '<nnnnabledgfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgf>',
        '<nnnnnnabledgfabledgfablennfabledgfnnfabledgf>',
        '<nnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgf>',
        '<nnnnnnnnnabledgfabledgfabledgfnnfabledgfnnfabledgfnnfabledgf>'
        ]

compressor.compress_dataset(dataset)

likelihoods = []
for alpha in [0.001, 0.1, 1, 10, 100]:
    print(f'alpha: {alpha}')
    compressor = ProbZip(alpha=alpha)
    compressor.compress_dataset(dataset)
    likelihood = compressor.get_dataset_likelihood(dataset)
    likelihoods.append(likelihood)

plt.plot(likelihoods)
plt.show()
