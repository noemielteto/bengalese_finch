from bengalese_finch.models.probzip import *
from collections import defaultdict
import numpy as np
import pandas as pd
import random

data = get_data(experimenter='Lena', strings=True)
subject_dataset = data['wh09pk88']
dataset = subject_dataset

def get_transmat(symbols, dataset):
    transitions = defaultdict(lambda: defaultdict(int))
    
    expanded_symbols = expand_symbols(symbols)
    
    # Process dataset to count transitions in terms of expanded symbols
    for sequence in dataset:
        mapped_sequence = []
        start = 0
        while start < len(sequence):
            matches = []
            for key, value in expanded_symbols.items():
                end = min(len(sequence), start + len(value))
                if sequence[start:end] == value:
                    matches.append(key)
            best_match = max(matches, key=lambda x: len(expanded_symbols[x]), default=None)
            mapped_sequence.append(best_match)
            start += len(expanded_symbols[best_match])
        
        # Count symbol transitions
        for i in range(len(mapped_sequence) - 1):
            from_symbol = mapped_sequence[i]
            to_symbol = mapped_sequence[i + 1]
            transitions[from_symbol][to_symbol] += 1
    
    # Convert counts to probabilities
    transmat = pd.DataFrame(index=symbols.keys(), columns=symbols.keys(), data=0.0)
    for symbol in transitions:
        total = sum(transitions[symbol].values())
        for next_symbol in transitions[symbol]:
            transmat.at[symbol, next_symbol] = transitions[symbol][next_symbol] / total
    
    return transmat


def get_new_concat_symbols(symbols, dataset, threshold=.99, binary=True):

    transmat = get_transmat(symbols, dataset)

    max_n = max([int(k.split('_')[1]) for k in transmat.index if k.startswith('X_')], default=-1)
    new_symbols = {}

    for symbol in transmat.index:
        
        new_symbol = [symbol]
        next_symbol = symbol
        while True:
            next_candidates = transmat.loc[next_symbol]
            next_symbol = next_candidates.idxmax()
            
            if next_candidates[next_symbol] >= threshold:
                new_symbol.append(next_symbol)
                if binary:
                    break
            else:
                break

        if len(new_symbol) > 1:
            max_n += 1
            new_symbols[f'X_{max_n}'] = new_symbol
    
    return new_symbols

def get_new_repeat_symbols(symbols, dataset):
    new_symbols = {}
    max_n = max([int(k.split('_')[1]) for k in symbols.keys() if k.startswith('X_')], default=-1)
    expanded_symbols = expand_symbols(symbols)
    
    for sequence in dataset:
        start = 0
        while start < len(sequence):
            for key, value in expanded_symbols.items():
                repeat_length = 1
                while sequence[start:start + len(value) * (repeat_length + 1)] == value * (repeat_length + 1):
                    repeat_length += 1
                
                repeat_symbol = [value]*repeat_length
                if (repeat_length > 1) and (repeat_symbol not in new_symbols.values()) and (repeat_symbol not in symbols.values()):
                    max_n += 1
                    new_symbols[f'X_{max_n}'] = repeat_symbol
                
            start += 1
    
    return new_symbols

def get_new_symbols(symbols, dataset, threshold=0.99, binary=True):
    symbols_copy = symbols.copy()
    new_concat_symbols = get_new_concat_symbols(symbols_copy, dataset, threshold=threshold, binary=binary)
    symbols_copy.update(new_concat_symbols)  # Update symbols with new concatenated symbols so that they can be used in repeat symbol detection and repeats are labelled uniquely
    new_repeat_symbols = get_new_repeat_symbols(symbols_copy, dataset)
    new_symbols = {**new_concat_symbols, **new_repeat_symbols}

    return new_symbols

def get_terminals(dataset):
    terminals = {}
    n = 0
    flattened_dataset = ''.join(dataset)
    symbols = set(flattened_dataset)
    for symbol in symbols:
        terminals[f'X_{n}'] = symbol
        n += 1

    return terminals

def expand_symbols(symbols):
    expanded_symbols = {}
    
    def expand(value):
        """ Recursively expands a symbol, handling both lists of symbols and characters."""
        if isinstance(value, list):
            return "".join(expand(symbols.get(v, v)) for v in value)
        return symbols.get(value, value)  # Expand only if the value is in symbols
    
    for key, value in symbols.items():
        expanded_symbols[key] = expand(value)
    
    return expanded_symbols

def get_symbols(dataset, threshold=0.99, binary=True):

    symbols = get_terminals(dataset)
    new_symbols = get_new_symbols(symbols=symbols, dataset=dataset, threshold=threshold, binary=binary)

    while len(new_symbols) > 0:

        print(len(new_symbols))
        print(new_symbols)

        symbols.update(new_symbols)
        new_symbols = get_new_symbols(symbols=symbols, dataset=dataset, threshold=threshold, binary=binary)

    return symbols


# symbols = get_symbols(dataset)

def inside_outside_algorithm(symbols, dataset, max_iterations=10, beam_size=100, epsilon=1e-10):
    """
    Implements the Inside-Outside algorithm with beam search sampling to estimate probabilities.
    """
    expanded_symbols = expand_symbols(symbols)
    rule_counts = defaultdict(float)
    rule_probabilities = {key: 1.0 / len(symbols) for key in symbols}  # Initialize rule probabilities
    
    for _ in range(max_iterations):
        inside_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        outside_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        total_likelihood = 0.0
        
        for sequence in dataset:
            # n = len(sequence)
            n = 10
            sampled_lengths = random.sample(range(1, n + 1), min(beam_size, n))  # Sample substrings
            
            # Compute Inside probabilities β(X, i, j)
            for length in sampled_lengths:
                print(f'length: {length}')
                for i in range(n - length + 1):
                    j = i + length - 1
                    for X, value in expanded_symbols.items():
                        if sequence[i:j+1] == value:
                            inside_prob[X][i][j] = rule_probabilities[X]
                        else:
                            for k in range(i, j):
                                for Y in expanded_symbols:
                                    for Z in expanded_symbols:
                                        if inside_prob[Y][i][k] > 0 and inside_prob[Z][k+1][j] > 0:
                                            inside_prob[X][i][j] += rule_probabilities[X] * inside_prob[Y][i][k] * inside_prob[Z][k+1][j]
            
            total_likelihood += inside_prob['S'][0][n-1] if inside_prob['S'][0][n-1] > epsilon else epsilon  # Avoid zero likelihood
            print(f'total_likelihood: {total_likelihood}')
            
            # Compute Outside probabilities α(X, i, j)
            outside_prob['S'][0][n-1] = 1.0  # Initialize root
            for length in reversed(sampled_lengths):
                for i in range(n - length + 1):
                    j = i + length - 1
                    for X in expanded_symbols:
                        for k in range(i, j):
                            for Y in expanded_symbols:
                                for Z in expanded_symbols:
                                    if inside_prob[Y][i][k] > 0 and inside_prob[Z][k+1][j] > 0 and inside_prob[X][i][j] > epsilon:
                                        outside_prob[Y][i][k] += (outside_prob[X][i][j] * rule_probabilities[X] * inside_prob[Z][k+1][j]) / inside_prob[X][i][j]
                                        outside_prob[Z][k+1][j] += (outside_prob[X][i][j] * rule_probabilities[X] * inside_prob[Y][i][k]) / inside_prob[X][i][j]
            
            # Expectation step: compute expected counts for each rule
            for X in expanded_symbols:
                for i in range(n):
                    for j in range(i, n):
                        for k in range(i, j):
                            for Y in expanded_symbols:
                                for Z in expanded_symbols:
                                    if inside_prob[Y][i][k] > 0 and inside_prob[Z][k+1][j] > 0 and inside_prob['S'][0][n-1] > epsilon:
                                        prob = (outside_prob[X][i][j] * rule_probabilities[X] * inside_prob[Y][i][k] * inside_prob[Z][k+1][j]) / inside_prob['S'][0][n-1]
                                        rule_counts[X] += prob
        
        # Maximization step: update rule probabilities
        total_counts = sum(rule_counts.values())
        for X in rule_probabilities:
            rule_probabilities[X] = rule_counts[X] / total_counts if total_counts > epsilon else epsilon
    
    return rule_probabilities

# symbols = get_symbols(dataset)
rule_probabilities = inside_outside_algorithm(symbols, dataset)