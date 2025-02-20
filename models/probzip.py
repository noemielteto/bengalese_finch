from bengalese_finch.models.utils import *
import numpy as np
import math
from scipy.stats import poisson

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def rewrite_list(lst):
    if len(lst) == 0:
        return lst  # Return the empty list if it's empty

    # Check if all elements in the list are the same
    if all(x == lst[0] for x in lst):
        return [lst[0]] * len(lst)
    else:
        return lst  # Return the original list if not all elements are the same


def is_repeat(elements):
    return True if len(set(elements)) == 1 else False


def get_expression(parent=None, prefix=None, suffix=None, rate=None):

    if parent is None:
        expression = ''

    else:
        assert sum(x is not None for x in [prefix, suffix, rate]) == 1, "Exactly one of prefix, suffix, or rate must be defined."

        affix = prefix if prefix is not None else suffix

        if parent.is_emptystring():
            assert isinstance(affix, (str, type(None))), "Terminal nodes should have epsilon empty string as their parent."
            expression = affix

        else:

            if prefix is not None:

                a = f"'{prefix.expression}'" if prefix.type == 'terminal' else prefix.expression
                b = f"'{parent.expression}'" if parent.type == 'terminal' else parent.expression
                expression  = f'[{a}, {b}]'

            elif suffix is not None:

                a = f"'{parent.expression}'" if parent.type == 'terminal' else parent.expression
                b = f"'{suffix.expression}'" if suffix.type == 'terminal' else suffix.expression
                expression = f'[{a}, {b}]'

            else:
                if parent.type=='terminal':
                    expression = f'{[parent.expression]}*{rate}'
                else:
                    expression = f'{parent.expression}*{rate}'
    
    return expression


def get_flat_expression(parent=None, prefix=None, suffix=None, rate=None):

    if parent is None:
        flat_expression = ''
    
    else:
        assert sum(x is not None for x in [prefix, suffix, rate]) == 1, "Exactly one of prefix, suffix, or rate must be defined."
        affix = prefix if prefix is not None else suffix

        if parent.is_emptystring():
            assert isinstance(affix, (str, type(None))), "Terminal nodes should have epsilon empty string as their parent."
            flat_expression = affix

        else:

            if prefix is not None:
                flat_expression  = prefix.flat_expression + parent.flat_expression

            elif suffix is not None:
                flat_expression  = parent.flat_expression + suffix.flat_expression

            else:
                if parent.type=='terminal':
                    flat_expression  = parent.flat_expression * rate
                else:
                    flat_expression  = parent.flat_expression * rate
        
    return flat_expression

class Node:

    def __init__(self, alpha, parent=None, prefix=None, suffix=None, affix=None, rate=None):

        self.alpha          = alpha
        self.parent         = parent
        self.prefix         = prefix
        self.suffix         = suffix
        self.affix          = affix
        self.rate           = rate
        self.children       = []
        self.count          = 1

        if parent is None or parent.is_emptystring():
            self.type            = 'terminal'
        else:
            self.type            = 'nonterminal'
            
        self.expression = get_expression(parent=parent, prefix=prefix, suffix=suffix, rate=rate)
        self.flat_expression = get_flat_expression(parent=parent, prefix=prefix, suffix=suffix, rate=rate)

        self.affix = prefix if prefix is not None else suffix
        
        if parent is not None:
            parent.children.append(self)

    def __repr__(self):
        return str(self.expression)

    def is_emptystring(self):
        return self.expression == ''

    def expand(self, rate=None):

        if self.type=='terminal':
            return self.expression

        else:
            if self.rate is None:
                return self.flat_expression

            else:
                # sampled repeat
                if rate is None:
                    rate = poisson.rvs(self.rate)

                return self.parent.flat_expression * rate

    # TODO(noemielteto): Remove in case we indeed only need to compute the likelihood of the data under the most likely 
    # def get_likely_completions(self, child, p_threshold=0.05):

    #     # return the empty string or the terminal
    #     if self == self.compressor.epsilon or self.parent == self.compressor.epsilon:
    #         return [self.expression]

    #     elif self.rate == None:
    #         return [self.affix.flat_expression]

    #     else:
    #         rates = np.array(range(2,16))
    #         pmf = poisson.pmf(rates, self.rate)
    #         likely_rates = rates[np.where(pmf>p_threshold)]
    #         return [self.expand(r) for r in likely_rates]

    def get_observed_repeat(self, data, i):

        observed_repeat = 0
        if self.flat_expression == '':
            return observed_repeat, i
        string = self.flat_expression * (observed_repeat+1)
        while string == data[i:i+len(string)]:
            observed_repeat += 1
            string = self.flat_expression * (observed_repeat+1)
        return observed_repeat

    def get_candidate_children(self, data, i):

        candidate_children = []
        if len(self.children)==0:
            return candidate_children

        observed_repeat = self.get_observed_repeat(data, i)
        for child in self.children:

            if child.rate is not None and observed_repeat:
                candidate_children.append(child)
        
            else:
                # Get completion of empty string, terminal or nonterminal
                if child.expression == '' or child.parent.expression == '' or child.affix is None:
                    completion = child.expression
                else:
                    completion = child.affix.flat_expression
                
                # Mask the completion with the data -- only consider children that match the data
                if completion == data[i:i+len(completion)]:
                    candidate_children.append(child)

        return candidate_children

    def get_poisson_weights(self, data, i, children):

        poisson_weights = []
        observed_repeat = self.get_observed_repeat(data, i)
        for child in children:
            if child.rate is not None:
                p_poisson = poisson.pmf(observed_repeat+1, child.rate) + (1-poisson.pmf(child.rate, child.rate))
            else:
                p_poisson = 1
            poisson_weights.append(p_poisson)
        
        return poisson_weights

    def get_shift(self, data, i):

        if self.rate is not None:
            observed_repeat = self.get_observed_repeat(data, i)
            shift = len(self.expand(rate=(observed_repeat+1)))-1
        else:
            if self.type == 'terminal':
                shift = len(self.flat_expression)
            else:
                shift = len(self.affix.flat_expression)
        return shift

    def update(self, data, i):

        children = self.get_candidate_children(data, i)

        # no candidates -> stay
        if not len(children):
            return self, i

        counts      = np.array([child.count for child in children])
        weights     = self.get_poisson_weights(data, i, children)
        counts      = counts * weights
        N           = counts.sum()
        alpha       = self.alpha
        norm        = N + alpha
        prob_stay   = alpha/norm

        # We never infer the empty string because we have to move it at least one character to parse
        # TODO(noemielteto) Consider removing this and let the algorithm decide whether to stay or not?
        if (np.random.random() < prob_stay) and (self.expression != ''):
            self.count += 1
            return self, i

        else:
            N_children = counts.sum()
            norm_children = N_children
            probs_seat = counts / norm_children
            # print(f'N: {N}, alpha: {alpha}, norm: {norm}, prob_stay: {prob_stay}, probs: {probs}')
            child = np.random.choice(children, p=probs_seat)
            shift = child.get_shift(data, i)

            child, i = child.update(data, i+shift)

        return child, i

    # def infer(self, data, i):

    #     if i>=len(data):
    #         return self

    #     children = self.get_candidate_children(data, i)

    #     # no candidates -> stay
    #     if not len(children):
    #         return self, i

    #     else:
    #         counts  = np.array([child.count for child in children])
    #         weights = self.get_poisson_weights(data, i, children)
    #         counts = counts * weights
    #         # Instead of sampling, get the most likely child
    #         child  = children[np.argmax(counts)]
    #         shift = child.get_shift(data, i)
    #         child, i = child.infer(data, i+shift)

    #     return child, i

    def probability_compress(self, data, i):
        
        if self.expression == '':
            prob = 1

        else:
            # siblings = self.parent.get_candidate_children(data, i)
            siblings = self.parent.children
            counts   = np.array([sibling.count for sibling in siblings])
            weights     = self.get_poisson_weights(data, i, siblings)
            counts      = counts * weights
            N        = counts.sum()
            norm     = N + self.alpha
        
            if self.parent.expression == '':
                prob = (self.count / norm)
            else:
                prob = (self.count / norm) * self.parent.probability_compress(data, i)

        return prob

    def probability_not_compress(self, data, i):

        # children = self.get_candidate_children(data, i)
        children = self.children
        if len(children):
            counts = np.array([child.count for child in children])
            weights = self.get_poisson_weights(data, i, children)
            counts = counts * weights
            N = counts.sum()
            alpha = self.alpha
            norm = N + alpha
            return (alpha / norm)
        else:
            return 1

    def probability(self, data, i):
        
        prob_compress = self.probability_compress(data, i)
        prob_not_compress = self.probability_not_compress(data, i)

        return prob_compress * prob_not_compress


class ProbZip:

    def __init__(self, alpha=1):
        self.alpha                  = alpha
        self.epsilon                = Node(alpha=self.alpha)
        self.library                = {'': self.epsilon}

    def __repr__(self):

        sorted_items = sorted(self.library.items(), key=lambda item: item[1].count, reverse=True)

        r = f'{self.__class__.__name__}(\n'
        # Format each key-value pair and append it to the result string
        for node_expression, node in sorted_items:
            r += f"    {node_expression!r}: {node.count!r},\n"
        # Close the curly brace
        r += ')'

        return r

    def get_terminals(self, data):

        # memoize terminals
        terminals = set(data)
        self.n_terminals = len(terminals)
        for terminal in terminals:
            node = Node(alpha=self.alpha, parent=self.epsilon, suffix=terminal)
            self.library[terminal] = node

    def get_important_library(self, threshold=.95):
        total_count = sum([node.count for node in self.library.values()])
        sorted_library_items = sorted(self.library.items(), key=lambda x: x[1].count, reverse=True)
        i = 0
        important_library = {}
        while sum([node.count for node in important_library.values()])/total_count < threshold:
            expression, node = sorted_library_items[i]
            important_library[expression] = node
            i += 1

        return important_library

    def compress(self, data):

        i = 0
        parent, i = self.epsilon.update(data, i)
        while i<len(data):
            
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
            else:
                suffix, i = self.epsilon.update(data, i)
                rate = None
            
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            if expression in self.library.keys():
                child = self.library[expression]
                child.count += 1
            else:
                child = Node(alpha=self.alpha, parent=parent, suffix=suffix, rate=rate)
                self.library[expression] = child

            parent = child
        
        return parent

    def get_library_leaves(self):
        return [node for node in self.library.values() if len(node.children)==0]
    
    def get_shannon_entropy(self, data):
        leaves = self.get_library_leaves()
        leaf_probs = [leaf.probability(data, 0) for leaf in leaves]
        entropy = 0.0
        for p in leaf_probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def get_dataset_shannon_entropy(self, dataset):
        entropies = [self.get_shannon_entropy(data) for data in dataset]
        return np.mean(entropies)

    # Compress whole bouts; This approach suffers from greedyness
    # TODO(noemielteto): Remove in case we stick to the random substring approach
    # def compress_dataset(self, dataset):
    #     self.get_terminals(flatten_arbitrarily_nested_lists(dataset))
    #     for song in dataset:
    #         self.compress(song)

    # Compress random substrings of songs; This approach is more robust
    def compress_dataset(self, dataset, steps=1000):
        self.get_terminals(flatten_arbitrarily_nested_lists(dataset))
        for _ in range(steps):
            song = np.random.choice(dataset)
            # random substring of song
            i, j = sorted(np.random.choice(range(len(song)), 2))
            if i==j:
                continue
            substring = song[i:j]
            _ = self.compress(substring)

    def get_dataset_ll(self, dataset):
        liks = []
        for data in dataset:
            symbol = self.compress(data)
            liks.append(symbol.probability(data, 0))
        return np.sum(np.log((np.array(liks))))
    

    # def plot(self, save_name=None, threshold=.95):

    #     important_library = self.get_important_library(threshold=threshold)

    #     nodes = list(important_library.keys())
    #     terminals = [x for x in nodes if len(x)==1]
    #     nonterminals = [x for x in nodes if len(x)>1]

    #     edges = []
    #     for node_expression, node in important_library.items():

    #         if not len(important_library[node_expression].children):
    #             # TODO: mark terminal?
    #             continue

    #         children = important_library[node_expression].children
    #         # left child only
    #         # children = [important_library[node_expression].children[0]]

    #         for child in children:

    #             # weighted
    #             # edges.append((child.expression, node_expression, compressor.library[child.expression].parents[node]))
    #             # unweighted
    #             edges.append((child.expression, node_expression))

    #     # Create a directed graph
    #     G = nx.DiGraph()

    #     for edge in edges:
    #         # G.add_edge(edge[0], edge[1], weight=edge[2])
    #         G.add_edge(edge[0], edge[1])

    #     # weights = [G[u][v]['weight'] for u, v in G.edges()]
    #     # normalized_weights = [w / max(weights) * 10 for w in weights]

    #     # pos = graphviz_layout(G, prog='dot')
    #     pos = tree_layout(G, root='')  # Root of tree is the empty string

    #     # only color nonterminals; terminals will stay black
    #     cmap = plt.cm.get_cmap('tab20', len(nonterminals))
    #     colors = [cmap(i) for i in range(len(nonterminals))]
    #     colormap = dict(zip(nonterminals, colors))

    #     _, ax = plt.subplots(figsize=(16, 4))
    #     # nx.draw_networkx_edges(G, pos, width=normalized_weights, ax=ax)  # Set edge widths based on weights
    #     nx.draw_networkx_edges(G, pos, ax=ax)
    #     nx.draw_networkx_labels(G, pos, font_color='white', ax=ax)

    #     # Draw nodes and edges separately to customize their properties
    #     # nx.draw_networkx_nodes(G, pos, node_color='white')
    #     rect_width = 4
    #     rect_height = 40
    #     for node, (x, y) in pos.items():
    #         color = colormap.get(node, 'black')  # Default to black if node color not specified

    #         scaling_factor = max(3, len(node))
    #         width = rect_width*scaling_factor  # increase minimum width so that boxes for terminals are not too narrow
    #         height = rect_height

    #         rectangle = mpatches.FancyBboxPatch((x - width / 2, y - height / 2), width, height,
    #                                         boxstyle="square,pad=0",
    #                                         ec='black', fill=True, facecolor=color, lw=3, transform=ax.transData)
    #         ax.add_patch(rectangle)

    #     plt.gca().invert_yaxis()
    #     plt.axis('off')
    #     plt.tight_layout()
    #     if save_name is not None:
    #         plt.savefig(save_name, dpi=500)
    #         plt.close('all')
    #     else:
    #         plt.show()


def tree_layout(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    Compute the positions for a hierarchical layout of a tree or DAG.

    Parameters:
    - G: networkx graph (should be a tree or DAG).
    - root: the root node of the current branch.
    - width: horizontal space allocated for this branch.
    - vert_gap: gap between levels of hierarchy.
    - vert_loc: vertical location of the root.
    - xcenter: horizontal location of the root.
    - pos: dictionary of positions (used in recursion).
    - parent: parent of the current root (to avoid revisiting in undirected graphs).

    Returns:
    - pos: A dictionary mapping each node to its (x, y) position.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    # Get neighbors; for undirected graphs, remove the parent to avoid going backwards.
    neighbors = list(G.neighbors(root))
    if parent is not None and parent in neighbors:
        neighbors.remove(parent)

    if len(neighbors) != 0:
        # Divide the horizontal space among children.
        dx = width / len(neighbors)
        next_x = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            next_x += dx
            pos = tree_layout(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos, parent=root)
    return pos