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

        if parent.is_emptystring() and isinstance(affix, str):
            expression = affix

        elif parent.is_emptystring() and isinstance(affix, Node):
            expression = affix.expression

        elif affix is not None:
            if affix.is_emptystring() and isinstance(parent, Node):
                expression = parent.expression

            else:
                if prefix is not None:
                    if isinstance(prefix, str):
                        a = prefix
                    else:
                        a = f"'{prefix.expression}'" if prefix.type == 'terminal' else prefix.expression

                    if isinstance(parent, str):
                        b = parent
                    else:
                        b = f"'{parent.expression}'" if parent.type == 'terminal' else parent.expression
                    expression  = f'[{a}, {b}]'

                elif suffix is not None:

                    if isinstance(suffix, str):
                        a = suffix
                    else:
                        a = f"'{suffix.expression}'" if suffix.type == 'terminal' else suffix.expression

                    if isinstance(parent, str):
                        b = parent
                    else:
                        b = f"'{parent.expression}'" if parent.type == 'terminal' else parent.expression
                    expression  = f'[{b}, {a}]'

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

        # TODO(noemielteto) remove this and above when clear on what to do with empty strings
        if False:
            pass
        # if parent.is_emptystring():
        #     assert isinstance(affix, (str, type(None))), "Terminal nodes should have epsilon empty string as their parent."
        #     flat_expression = affix

        else:

            if prefix is not None:
                if isinstance(prefix, str):
                    flat_expression  = prefix + parent.flat_expression
                else:
                    flat_expression  = prefix.flat_expression + parent.flat_expression

            elif suffix is not None:
                if isinstance(suffix, str):
                    flat_expression  = parent.flat_expression + suffix
                else:
                    flat_expression  = parent.flat_expression + suffix.flat_expression

            else:
                if parent.type=='terminal':
                    flat_expression  = parent.flat_expression * rate
                else:
                    flat_expression  = parent.flat_expression * rate
        
    return flat_expression

class Node:

    def __init__(self, alpha_vector, parent=None, prefix=None, suffix=None, affix=None, rate=None):

        self.parent         = parent
        self.prefix         = prefix
        self.suffix         = suffix
        self.affix          = affix
        self.rate           = rate
        self.children       = []
        self.count          = 1

        if parent is None:
            self.type            = 'terminal'
            self.order           = 0        
        elif parent.is_emptystring():
            self.type            = 'terminal'
            self.order           = 1
        else:
            self.type            = 'nonterminal'
            if rate is None:
                self.order           = parent.order + suffix.order
            else:
                self.order           = parent.order + 1

        self.alpha = alpha_vector[self.order]

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

    def get_observed_repeat(self, data, i):

        observed_repeat = 0
        if self.flat_expression == '':
            return observed_repeat
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
                # TODO: remove this when clear
                # if child.expression == '' or child.parent.expression == '' or child.affix is None:
                #     completion = child.expression
                if child.type == 'terminal':
                    completion = child.flat_expression
                elif child.affix is None:
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
            observed_repeat = self.parent.get_observed_repeat(data, i)
            shift = len(self.expand(rate=observed_repeat+1))-1
        else:
            if self.type == 'terminal':
                shift = len(self.flat_expression)
            else:
                shift = len(self.affix.flat_expression)
        return shift

    def infer(self, data, i, update_counts=True):

        children = self.get_candidate_children(data, i)

        # print('---------------------------------------')
        # print('Candidate children:', children)

        # no candidates -> stay
        if not len(children):
            # print('No candidates. Staying.')
            return self, i

        counts      = np.array([child.count for child in children])
        weights     = self.get_poisson_weights(data, i, children)
        counts      = counts * weights
        N           = counts.sum()
        alpha       = self.alpha
        norm        = N + alpha
        prob_stay   = alpha/norm

        # NOTE: There is a decision here on whether we allow inferring empty string or not.
        if (np.random.random() < prob_stay) and (self.expression != ''):
        # if (np.random.random() < prob_stay):
            # print('Probabilistically staying.')
            if update_counts:
                self.count += 1
            return self, i

        else:
            N_children = counts.sum()
            norm_children = N_children
            probs_seat = counts / norm_children
            # print('Probs seat:', probs_seat)
            # print(f'N: {N}, alpha: {alpha}, norm: {norm}, prob_stay: {prob_stay}, probs: {probs}')
            child = np.random.choice(children, p=probs_seat)
            # print('Selected child:', child)
            shift = child.get_shift(data, i)

            child, i = child.infer(data=data, i=i+shift, update_counts=update_counts)

        return child, i

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
            alpha    = self.alpha
            norm     = N + alpha
        
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

    def get_predictive_distr(self):
        
        counts   = np.array([child.count for child in self.children])
        N        = counts.sum()
        alpha    = self.alpha
        norm     = N + alpha

        distr = list(counts/norm) + [alpha/norm]

        return np.array(distr)

class ProbZip:

    def __init__(self, alpha_vector):
        self.alpha_vector           = alpha_vector
        self.epsilon                = Node(alpha_vector=alpha_vector)
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

        data = flatten_arbitrarily_nested_lists(data)
        terminals = set(data)
        self.n_terminals = len(terminals)
        for terminal in terminals:
            node = Node(alpha_vector=self.alpha_vector, parent=self.epsilon, suffix=terminal)
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
    
    def compress_chain(self, data, update_counts=True):

        i = 0

        # Several steps
        while i<len(data):

            parent, i = self.epsilon.infer(data, i)
  
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
                i += len(parent.flat_expression) * observed_repeat
            else:
                suffix, i = self.epsilon.infer(data, i)
                rate = None
            
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            if expression in self.library.keys():
                child = self.library[expression]
            elif expression in [c.expression for c in parent.children]:
                c_i = 0
                while parent.children[c_i].expression != expression:
                    c_i += 1
                child = parent.children[c_i]
            else:
                child = Node(alpha_vector=self.alpha_vector, parent=parent, suffix=suffix, rate=rate)

            if update_counts:
                norm = parent.alpha + child.count
                prob_stay = parent.alpha/norm

                # Note: We prevent the creation of overlapping nonterminals here: if its parent
                # or affix is already in the library, we don't memoize it. If its suffix is parent
                # in the library, we don't memoize it.
                if np.random.random() > prob_stay and not len(self.get_nonterminal_overlaps(child)) and child.order<7:

                    # We only memoize non-reduntant expressions
                    if child.flat_expression not in [node.flat_expression for node in self.library.values()]:

                        # Memoize in library
                        if expression in self.library.keys():
                            child.count += 1
                        else: 
                            print('Memoizing expression:', expression)
                            self.library[expression] = child

                    else:
                        del child
                
                else:
                    del child
                
            else:
                del child
        
        return None

    # TODO: unify compress and compress_onestep
    def compress_onestep(self, data, update_counts=True):

        i = 0

        # Step 0
        parent, i = self.epsilon.infer(data, i)

        # Step 1 (only if there is more to infer)
        if i<(len(data)-1):
  
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
            else:
                suffix, i = self.epsilon.infer(data, i)
                rate = None
            
            # print(f'Parent: {parent}, Suffix: {suffix}, Rate: {rate}')
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            # TODO(noemielteto): rewrite this in terms of infer()

            if expression in self.library.keys():
                child = self.library[expression]
            elif expression in [c.expression for c in parent.children]:
                c_i = 0
                while parent.children[c_i].expression != expression:
                    c_i += 1
                child = parent.children[c_i]
            else:
                child = Node(alpha_vector=self.alpha_vector, parent=parent, suffix=suffix, rate=rate)

            if update_counts:
                norm = parent.alpha + child.count
                prob_stay = parent.alpha/norm

                # Note: We prevent the creation of overlapping nonterminals here: if its parent
                # or affix is already in the library, we don't memoize it. If its suffix is parent
                # in the library, we don't memoize it.
                if np.random.random() > prob_stay and not len(self.get_nonterminal_overlaps(child)) and child.order<7:

                    # We only memoize non-reduntant expressions
                    if child.flat_expression not in [node.flat_expression for node in self.library.values()]:

                        # Memoize in library
                        if expression in self.library.keys():
                            child.count += 1
                        else: 
                            print('Memoizing expression:', expression)
                            self.library[expression] = child

                        parent = child

                    else:
                        del child
                
                else:
                    del child
                
            else:
                del child
        
        return parent

    def compress(self, data, update_counts=False):

        children_created = []

        i = 0
        parent, i = self.epsilon.infer(data, i, update_counts=False)
        while i<len(data):
            
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
            else:
                suffix, i = self.epsilon.infer(data, i, update_counts=False)
                rate = None
            
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            # TODO(noemielteto): rewrite this in terms of infer()

            if expression in self.library.keys():
                child = self.library[expression]
            elif expression in [c.expression for c in parent.children]:
                c_i = 0
                while parent.children[c_i].expression != expression:
                    c_i += 1
                child = parent.children[c_i]
            else:
                child = Node(alpha_vector=self.alpha_vector, parent=parent, suffix=suffix, rate=rate)

            children_created.append(child)
            parent = child
        
        return parent, children_created

    
    def random_node_to_add(self):

        parent = np.random.choice(list(self.library.values()))
        # Prior for suffix or rate is .5; Can be hyperparameter
        if np.random.random() < .5:
            suffix = np.random.choice(list(self.library.values()))
            rate = None
        else:
            rate = np.random.choice([1, 2, 3, 4, 5])
            suffix = None

        child = Node(alpha_vector=self.alpha_vector, parent=parent, suffix=suffix, rate=rate)

        return child

    def random_node_to_remove(self):

        # Should we remove leaves only? Or entire subtrees?
        # node = np.random.choice(list(self.library.values()))

        leaves = [node for node in self.library.values() if len(node.children)==0]
        node = np.random.choice(leaves)
        
        return node

    def search_add_remove(self, dataset_train, dataset_val, steps=1000, log_every=100):

        self.get_terminals(dataset_train)
        results_dict = {'ll_train': [], 'll_val': [], 'entropy': []}

        for step in range(steps):

            ll_0 = self.get_dataset_ll(dataset_train)
            node_to_add = self.random_node_to_add()
            print('Considering adding node:', node_to_add)
            ll = self.get_dataset_ll(dataset_train)
            if ll > ll_0:
                self.library[node_to_add.expression] = node_to_add
                ll_0 = ll
                print('Added node:', node_to_add)
                print('Node order:', node_to_add.order)
                print('LL:', ll)
            else:
                del node_to_add
            
            node_to_remove = self.random_node_to_remove()
            print('Considering removing node:', node_to_remove)
            child_to_remove = None
            # This part would be much more straightforward if children was a dictionary
            for child in node_to_remove.parent.children:
                if child.expression in node_to_remove.expression:
                    child_to_remove = child
                    break

            del child_to_remove
            ll = self.get_dataset_ll(dataset_train)
            if ll > ll_0:
                del self.library[node_to_remove.expression]
                ll_0 = ll
                print('Removed node:', node_to_remove)
                print('LL:', ll)
            else:
                node_to_remove.parent.children.append(node_to_remove)

            if step % log_every == 0:
                
                ll_train = self.get_dataset_ll(dataset_train)
                ll_train /= len(flatten_arbitrarily_nested_lists(dataset_train))
                results_dict['ll_train'].append(ll_train)
                ll_val = self.get_dataset_ll(dataset_val)
                ll_val /= len(flatten_arbitrarily_nested_lists(dataset_val))
                results_dict['ll_val'].append(ll_val)

                entropy = self.get_shannon_entropy()
                entropy /= len(flatten_arbitrarily_nested_lists(dataset_train))
                results_dict['entropy'].append(entropy)
        
                print(f'Step {step}: normalized ll train: {ll_train}, normalized ll val: {ll_val}, normalized entropy: {entropy}, library size: {len(self.library)}')

        return results_dict



    # def get_library_leaves(self):
    #     return [node for node in self.library.values() if len(node.children)==0]
    
    # def get_shannon_entropy(self, data):
    #     leaves = self.get_library_leaves()
    #     leaf_probs = [leaf.probability(data, 0) for leaf in leaves]
    #     entropy = 0.0
    #     for p in leaf_probs:
    #         if p > 0:
    #             entropy -= p * math.log2(p)
    #     return entropy

    # def get_dataset_shannon_entropy(self, dataset):
    #     entropies = [self.get_shannon_entropy(data) for data in dataset]
    #     return np.mean(entropies)

    def get_shannon_entropy(self):
        entropy = 0.0
        for node in self.library.values():
            probs = node.get_predictive_distr()
            entropy -= np.sum(probs * np.log2(probs))
        return entropy

    def remove_redundant_nodes(self):
        n_redundant = 0
        for key1, node1 in self.library.items():
            for key2, node2 in self.library.items():
                if key1 != key2:
                    if node1.flat_expression == node2.flat_expression:
                        del self.library[key2]
                        n_redundant += 1
        return n_redundant

    def get_nonterminal_overlaps(self, node):
        overlaps = []
        for key, other_node in self.library.items():
            if node.parent == other_node.affix or node.affix == other_node.parent:
                overlaps.append(other_node)
        return overlaps

    def compress_dataset(self, dataset_train, dataset_val=None, steps=1000, log_every=100):

        results_dict = {'ll_train': [], 'll_val': [], 'entropy': []}
        n_train = len(flatten_arbitrarily_nested_lists(dataset_train))
        n_val = len(flatten_arbitrarily_nested_lists(dataset_val)) if dataset_val is not None else None

        self.get_terminals(dataset_train)
        for step in range(steps):
            song = np.random.choice(dataset_train)
            # random start index in song
            i = np.random.randint(len(song))
            # i = 0
            _ = self.compress_onestep(song[i:])
            # _ = self.compress_chain(song)
            # _, _ = self.compress(song)

            if step % log_every == 0:

                ll_train = self.get_dataset_ll(dataset_train)
                ll_train /= n_train
                results_dict['ll_train'].append(ll_train)
                ll_val = None

                entropy = self.get_shannon_entropy()
                entropy /= n_train
                results_dict['entropy'].append(entropy)

                if dataset_val is not None:
                    ll_val = self.get_dataset_ll(dataset_val)
                    ll_val /= n_val
                    results_dict['ll_val'].append(ll_val)
        
                print(f'Step {step}: normalized ll train: {ll_train}, normalized ll val: {ll_val}, normalized entropy: {entropy}, library size: {len(self.library)}')

        return results_dict

    def get_dataset_ll(self, dataset):
        liks = []
        for data in dataset:
            symbol, children_created = self.compress(data)
            liks.append(symbol.probability(data, 0))

            # clean up inferred children!
            for child in children_created:
                del child
            
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