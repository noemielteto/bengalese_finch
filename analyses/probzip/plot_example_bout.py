from probzip.probzip import *

subject = 'wh09pk88'
alpha = 0.00100000

data = get_data(experimenter='Lena', strings=True)
dataset = data[subject]


entropies = []
for alpha in [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
# for alpha in [10, 100, 1000]:    
    print(f'alpha: {alpha}')
    compressor = ProbZip(alpha=alpha)
    compressor.compress_dataset(dataset=dataset, steps=1000)
    entropy = compressor.get_dataset_shannon_entropy(dataset)
    entropies.append(entropy)

# compressor = ProbZip(alpha=alpha)
# compressor.compress_dataset(dataset=dataset, steps=1000)

# len(compressor.library)

# symbol = compressor.compress(dataset[0])

# compressor.plot(threshold=.01)


# len(symbol.parent.parent.parent.parent.parent.parent.parent.children)