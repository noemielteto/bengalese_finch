from bengalese_finch.models.utils import *
import numpy as np
from scipy.special import gammaln  # log-factorial
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

class ZeroTruncatedPoisson:
    def __init__(self):
        pass  # no need for max_k anymore

    def pmf(self, k, mu):
        k = np.asarray(k)
        mu = float(mu)
        pmf_vals = np.zeros_like(k, dtype=float)
        valid = k >= 1
        k_valid = k[valid]
        log_pmf = -mu + k_valid * np.log(mu) - gammaln(k_valid + 1)
        pmf_vals[valid] = np.exp(log_pmf) / (1 - np.exp(-mu))
        return pmf_vals if pmf_vals.shape else float(pmf_vals)

    def rvs(self, mu, random_state=None):
        """Draw a single sample from a zero-truncated Poisson."""
        rng = np.random.default_rng(random_state)
        while True:
            x = rng.poisson(mu)
            if x > 0:
                return x

    def mean(self, mu):
        mu = float(mu)
        return mu / (1 - np.exp(-mu))

    def var(self, mu):
        mu = float(mu)
        e_neg_mu = np.exp(-mu)
        denom = 1 - e_neg_mu
        return mu / denom * (1 + mu * e_neg_mu / denom) - (mu / denom)**2
zt_poisson = ZeroTruncatedPoisson()

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

    def __init__(self, alpha, parent=None, prefix=None, suffix=None, affix=None, rate=None):

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

        self.alpha = alpha

        self.expression = get_expression(parent=parent, prefix=prefix, suffix=suffix, rate=rate)
        self.flat_expression = get_flat_expression(parent=parent, prefix=prefix, suffix=suffix, rate=rate)

        self.affix = prefix if prefix is not None else suffix
        
        if parent is not None:
            parent.children.append(self)
            self.completions = parent.sample_completions(suffix, rate)

    def __repr__(self):
        return str(self.expression)

    def is_emptystring(self):
        return self.expression == ''

    def expand(self, sampled_repeat=False):

        # Recursively expand the expression

        if self.type == 'terminal':
            return self.expression
        elif self.rate is not None:
            if sampled_repeat:
                rate = zt_poisson.rvs(self.rate)
            else:
                rate = self.rate
            return [self.parent.expand(sampled_repeat=sampled_repeat)] * rate
        else:
            return [self.parent.expand(sampled_repeat=sampled_repeat), self.suffix.expand(sampled_repeat=sampled_repeat)]

    def sample_completions(self, suffix=None, rate=None, n_samples=10):
        
        # Similarly to expand(), recursively expands but only the right side of the expression
        # (that is, the suffix or the repetation of the parent), allowing to condition on a fixed parent.

        assert sum(x is not None for x in [suffix, rate]) == 1, "Exactly one of suffix or rate must be defined."

        # It is very inelegant to do rate is None and not None separately; Maybe we should have a function that does expand_completion(), analogously to expand(), to handle both
        if rate is not None:
            # The completion is rate-1 repeats of the parent (the whole expression will be rate repeats of the parent)
            completions = [flatten([self.expand(sampled_repeat=True)] * zt_poisson.rvs(rate-1)) for _ in range(n_samples)]

        else:
            if isinstance(suffix, str):
                completions = [suffix]
            elif suffix.type == 'terminal':
                # No need to sample
                completions = [suffix.expression]
            else:
                completions = [flatten(suffix.expand(sampled_repeat=True)) for _ in range(n_samples)]

        # Convert to strings that can be used to match the data
        completions = [''.join(completion) for completion in completions]
        # Select unique completions
        completions = list(set(completions))

        return completions

    #TODO(): Consider changing this to observed_rate in order to avoid confusion around rate=repeat+1
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

        for child in self.children:
                
            # Mask the completion with the data -- only consider children that have sampled completions that match the data
            for completion in child.completions:
                if completion == data[i:i+len(completion)]:
                    candidate_children.append(child)
                    break

        return candidate_children

    def get_poisson_weights(self, data, i, children):

        poisson_weights = []
        observed_repeat = self.get_observed_repeat(data, i)
        for child in children:
            if child.rate is not None:
                p_poisson = zt_poisson.pmf(observed_repeat+1, child.rate) + (1-zt_poisson.pmf(child.rate, child.rate))
            else:
                p_poisson = 1
            poisson_weights.append(p_poisson)
        
        return poisson_weights

    # def get_shift(self, data, i):
    #     # TODO: Modify this such that we get the shift for the poisson rate not the observed rate
    #     # It should also be unified with expand() and sample_completions() somehow
    #     # Maybe: complete(sampled=False) could do one completion from which we get the shift;
    #     # And sample_completions() will call complete(sampled=True) n_samples times

    #     if self.rate is not None:
    #         # observed_repeat = self.parent.get_observed_repeat(data, i)
    #         flat_expression = flatten(self.parent.expand())
    #         shift = len(flat_expression)-1
    #     else:
    #         if self.type == 'terminal':
    #             shift = len(self.flat_expression)
    #         else:
    #             shift = len(self.affix.flat_expression)

    #     return shift

    def get_shift(self, data, i):
        
        # Longest completion that matches the data
        # We do this because we want to condition on all the data that we attribute to the chosen child;
        # Otherwise we would create splitting
        sorted_completions = sorted(self.completions, key=len, reverse=True)
        for c in sorted_completions:
            if c == data[i:i+len(c)]:
                break

        return len(c)

    def infer(self, data, i, update_counts=True):

        children = self.get_candidate_children(data, i)
        # print(children)

        # no candidates -> stay
        if not len(children):
            if update_counts:
                self.count += 1
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
            if update_counts:
                self.count += 1
            return self, i

        else:
            N_children = counts.sum()
            norm_children = N_children
            probs_seat = counts / norm_children
            child = np.random.choice(children, p=probs_seat)
            shift = child.get_shift(data, i)

            child, i = child.infer(data=data, i=i+shift, update_counts=update_counts)

        return child, i

    def predict(self):

        # Predict is the same as infer; The two differences are:
        # 1. predict() does not get 'to see' the candidate children that actally match the data
        # 2. Relatedly, predict() does not see the observed repeat that matches the data
        # 3. predict() does not update counts
        children = self.children

        # No children -> 
        if not len(children):
            return self

        counts      = np.array([child.count for child in children])
        N           = counts.sum()
        alpha       = self.alpha
        norm        = N + alpha
        prob_stay   = alpha/norm

        # NOTE: There is a decision here on whether we allow predicting empty string or not.
        if (np.random.random() < prob_stay) and (self.expression != ''):
            return self

        else:
            N_children = counts.sum()
            norm_children = N_children
            probs_seat = counts / norm_children
            child = np.random.choice(children, p=probs_seat)

            child = child.predict()

        return child

    def probability_compress(self, data, i):
        
        # Note: Start symbol is handled as a special symbol whose marginal probability
        # is 1 (Just like that of epsilon.)
        if self.expression == '' or self.expression == '<':
            prob = 1

        else:
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

    def get_entropy(self):

        predictive_distr = self.get_predictive_distr()
        entropy = -np.sum(predictive_distr * np.log2(predictive_distr + 1e-100))

        return entropy
    
    # TODO: Track this in the library, just as order
    def get_hierarchy(self):
        symbol = self
        hierarchy = 0
        while symbol.parent is not None:
            hierarchy += 1
            symbol = symbol.parent
        return hierarchy

class ProbZip:

    def __init__(self, alpha):
        self.alpha                  = alpha
        self.epsilon                = Node(alpha=alpha)
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

        data = flatten(data)
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
    
    def add(self, node):

        # TODO(noemi): Should have asserts here making sure we don't have children of parents not in library and vice versa

        if node not in node.parent.children:
            node.parent.children.append(node)
        if node.expression not in self.library.keys():
            self.library[node.expression] = node

    def remove(self, node):

        # TODO(noemi): Should have asserts here making sure we don't have children of parents not in library and vice versa

        if node in node.parent.children:
            node.parent.children.remove(node)
        if node.expression in self.library.keys():
            del self.library[node.expression]

    def compress_chain(self, data, update_counts=True):

        memoized_children = []

        # Note: We condition on start symbol and start at index 1
        i = 1
        parent, i = self.library['<'].infer(data, i)

        # Several steps
        while i<len(data):

            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
                i += len(parent.flat_expression) * observed_repeat
            else:
                suffix, i = self.epsilon.infer(data, i)
                rate = None
            
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            new_child=False
            if expression in self.library.keys():
                child = self.library[expression]
            elif expression in [c.expression for c in parent.children]:
                c_i = 0
                while parent.children[c_i].expression != expression:
                    c_i += 1
                child = parent.children[c_i]
            else:
                child = Node(alpha=self.alpha, parent=parent, suffix=suffix, rate=rate)
                new_child=True

            if update_counts:
                norm = parent.alpha + child.count
                prob_stay = parent.alpha/norm

                if np.random.random() > prob_stay:

                    # We only memoize non-reduntant expressions
                    if child.flat_expression not in [node.flat_expression for node in self.library.values()]:

                        # Memoize in library
                        if expression in self.library.keys():
                            child.count += 1
                        else: 
                            # print('Memoizing expression:', expression)
                            self.library[expression] = child
                            memoized_children.append(child)

                    else:
                        if new_child:
                            self.remove(child)
                
                else:
                    if new_child:
                        self.remove(child)
                
            else:
                if new_child:
                    self.remove(child)

            parent, i = self.epsilon.infer(data, i)
        
        return memoized_children
        

    # TODO: unify compress and compress_onestep
    def compress_onestep(self, data, update_counts=True):

        memoized_child = None

        i = 0

        # Step 0
        parent, i = self.epsilon.infer(data, i)

        # Step 1 (only if there is more to infer)
        if i<(len(data)-1):
  
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
                i += len(parent.flat_expression) * observed_repeat
            else:
                suffix, i = self.epsilon.infer(data, i)
                rate = None
            
            # print(f'Parent: {parent}, Suffix: {suffix}, Rate: {rate}')
            expression = get_expression(parent=parent, suffix=suffix, rate=rate)

            # TODO(noemielteto): rewrite this in terms of infer()

            new_child=False
            if expression in self.library.keys():
                child = self.library[expression]
            elif expression in [c.expression for c in parent.children]:
                c_i = 0
                while parent.children[c_i].expression != expression:
                    c_i += 1
                child = parent.children[c_i]
            else:
                child = Node(alpha=self.alpha, parent=parent, suffix=suffix, rate=rate)
                new_child=True

            if update_counts:
                norm = parent.alpha + child.count
                prob_stay = parent.alpha/norm

                if np.random.random() > prob_stay:

                    # We only memoize non-reduntant expressions
                    if child.flat_expression not in [node.flat_expression for node in self.library.values()]:

                        # Memoize in library
                        if expression in self.library.keys():
                            child.count += 1
                        else: 
                            # print('Memoizing expression:', expression)
                            self.library[expression] = child
                            memoized_child = child

                    else:
                        if new_child:
                            self.remove(child)
                
                else:
                    if new_child:
                        self.remove(child)
                
            else:
                if new_child:
                    self.remove(child)
        
        return memoized_child

    def compress(self, data, update_counts=False):

        memoized_children = []
        # Note: We condition on start symbol and start at index 1
        i = 1
        parent, i = self.library['<'].infer(data, i, update_counts=update_counts)

        while i<len(data):
            
            observed_repeat = parent.get_observed_repeat(data, i)
            if observed_repeat:
                rate = observed_repeat + 1
                suffix = None
                i += len(parent.flat_expression) * observed_repeat
            else:
                suffix, i = self.epsilon.infer(data, i, update_counts=update_counts)
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
                child = Node(alpha=self.alpha, parent=parent, suffix=suffix, rate=rate)
                memoized_children.append(child)

            parent = child

        return parent, memoized_children

    def compress_list(self, data, update_counts=True):

        symbols = []
        probs = []
        # Note: We condition on start symbol and start at index 1
        i = 1
        symbol, i_next = self.library['<'].infer(data, i, update_counts=update_counts)
        probs.append(symbol.probability(data, i))
        symbols.append(symbol)

        while i<len(data):
            
            i = i_next
            symbol, i_next = self.epsilon.infer(data, i, update_counts=update_counts)
            probs.append(symbol.probability(data, i))
            symbols.append(symbol)
            
        return symbols, probs

    def get_leaves(self):
        leaves = [node for node in self.library.values() if len(node.children)==0]
        return leaves

    # TODO: Rename
    def get_entropy(self):
        
        return np.sum([node.get_entropy() for node in self.library.values()])

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
        
        if node.rate is not None:
            return []

        overlaps = []
        for key, other_node in self.library.items():
            if node.parent == other_node.affix or node.affix == other_node.parent:
                overlaps.append(other_node)
        return overlaps

    def compress_dataset(self, dataset_train, dataset_val, dataset_test, steps=1000, prune_every=1000, log=False, log_every=100):

        results_dict = {'mdl_data_train': [],
                        'mdl_data_val': [],
                        'mdl_data_test': [],
                        'mdl_library': [],
                        'library_size': [],
                        'mdl': [],
                        'converged': None}

        n_train = len(flatten(dataset_train))
        n_val = len(flatten(dataset_val))
        n_test = len(flatten(dataset_test))

        self.get_terminals(dataset_train+dataset_val)
        # Important note: We do steps+1 steps to make sure that a final pruning happens at the end
        step = 0
        converged = False
        while not converged and step<steps:

            song = np.random.choice(dataset_train)

            # if True:
            # # Burn-in plus later alternation
            if (step<steps*.2) or (step % 2 == 0):
            # Alternation
            # if (step % 2 == 0):
                # t = time.time()
                i = np.random.randint(len(song))
                symbol = self.compress_onestep(song[i:])
                # print(f'Compress onestep took {time.time()-t} seconds.')
                # Note: We only remove newly memoized symbols; A more nuanced version will unseat customers but that will be much more costly as it would be done in every single step
                if symbol is not None:

                    # t = time.time()

                    mdl_1 = self.mdl([song])
                    self.remove(symbol)
                    mdl_0 = self.mdl([song])

                    if mdl_1 < mdl_0:
                        self.add(symbol)
                        #     print('Added symbol:', symbol)
                        # else:
                        #     print(f'Removed symbol: {symbol}')
                        # print(f'Checking whether to keep symbol took {time.time()-t} seconds.')

            # After burn-in, set all customer counts to 1
            elif step == steps*.2:
                for node in self.library.values():
                    node.count = 1
            
            else:
                # t = time.time()
                symbols = self.compress_chain(song)
                # print(f'Compress chain took {time.time()-t} seconds.')
                # _, symbols = self.compress(song, update_counts=False)
                # Note: We only remove newly memoized symbols; A more nuanced version will unseat customers but that will be much more costly as it would be done in every single step
                if len(symbols):

                    mdl_1 = self.mdl([song])

                    for symbol in symbols:
                        self.remove(symbol)

                    mdl_0 = self.mdl([song])
                    
                    if mdl_1 < mdl_0:
                        for symbol in symbols:
                            self.add(symbol)
                    
            if (step>steps*.2) and (step % prune_every == 0):
                self.prune(dataset_train)

            if log and (step % log_every == 0):

                mdl_data_train = - self.get_dataset_ll(dataset_train)
                mdl_data_train /= n_train
                results_dict['mdl_data_train'].append(mdl_data_train)

                mdl_data_val = - self.get_dataset_ll(dataset_val)
                mdl_data_val /= n_val
                results_dict['mdl_data_val'].append(mdl_data_val)

                mdl_data_test = - self.get_dataset_ll(dataset_test)
                mdl_data_test /= n_test
                results_dict['mdl_data_test'].append(mdl_data_test) 

                mdl_library = self.get_entropy()
                mdl_library /= n_val
                results_dict['mdl_library'].append(mdl_library)
                library_size = len(self.get_important_library(threshold=1))
                results_dict['library_size'].append(library_size)

                results_dict['mdl'].append(self.mdl(dataset_val)/n_val)

                print(f'Step {step}: normalized mdl train: {mdl_data_train}, normalized mdl val: {mdl_data_val}, normalized mdl test: {mdl_data_test}, normalized mdl library: {mdl_library}, library size: {library_size}')
                if log:
                    converged = self.converged(results_dict)
            else:
                if step % 1000 == 0:
                    print(f'Step {step}')
            
            step += 1

        # Final pruning
        self.prune(dataset_train)

        # Final MDL computation
        if not log:
            # TODO: Logging should be a function because it's repeated
            mdl_data_train = - self.get_dataset_ll(dataset_train)
            mdl_data_train /= n_train
            results_dict['mdl_data_train'].append(mdl_data_train)

            mdl_data_val = - self.get_dataset_ll(dataset_val)
            mdl_data_val /= n_val
            results_dict['mdl_data_val'].append(mdl_data_val)

            mdl_data_test = - self.get_dataset_ll(dataset_test)
            mdl_data_test /= n_test
            results_dict['mdl_data_test'].append(mdl_data_test) 

            mdl_library = self.get_entropy()
            mdl_library /= n_val
            results_dict['mdl_library'].append(mdl_library)
            library_size = len(self.get_important_library(threshold=1))
            results_dict['library_size'].append(library_size)

            results_dict['mdl'].append(self.mdl(dataset_val)/n_val)

        if log:
            results_dict['converged'] = converged
            print(f'Converged: {converged} after {step} steps.')
        else:
            print(f'{steps} steps completed.')
    
        return results_dict

    def prune(self, dataset):
        
        h = int(len(dataset)/2)
        dataset_1 = dataset[:h]
        dataset_2 = dataset[h:]

        for dataset in [dataset_1, dataset_2]:
            symbols_removed = True
            while symbols_removed:

                mdl_0 = self.mdl(dataset)
                leaves = self.get_leaves()
                nonterminal_leaves = [node for node in leaves if node.type=='nonterminal']
                symbols_to_remove = []
                for symbol in nonterminal_leaves:
                    self.remove(symbol)
                    mdl_1 = self.mdl(dataset)

                    if mdl_1 < mdl_0:
                        symbols_to_remove.append(symbol)
                    
                    self.add(symbol)

                # print(f'Removing symbols: {symbols_to_remove}')
                for symbol in symbols_to_remove:
                    self.remove(symbol)
                symbols_removed = len(symbols_to_remove)

    def unseat(self, dataset):
        
        symbols_to_unseat = []
        symbols_to_remove = []
        for _, symbol in self.library.items():

            # Terminal symbols are unremovable and have a nonzero probability
            if symbol.type == 'terminal' and symbol.count == 1:
                continue

            mdl_0 = self.mdl(dataset)
            symbol.count -= 1
            mdl_1 = self.mdl(dataset)
            symbol.count += 1

            if mdl_1 < mdl_0:
                symbols_to_unseat.append(symbol)
            
        for symbol in symbols_to_unseat:
            symbol.count -= 1

            if symbol.count == 0:
                # Nonterminals are removed when count is zero
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            self.remove(symbol)

    def converged(self, results_dict, threshold=0.01):
        """
        Check if the model has converged based on the change in MDL in last 10 recorded steps.
        """

        if len(results_dict['mdl']) < 10:
            return False

        # Calculate the differences between consecutive values
        differences = np.abs(np.diff(results_dict['mdl'][-10:]))

        # Threshold is threshold proportion of the last value
        threshold = threshold * np.abs(results_dict['mdl'][-1])

        # Check if all differences are below the threshold
        return np.all(differences < threshold)

    def generate(self):
        data = []
        symbol = self.library['<'].predict()
        data.append(symbol.expand(sampled_repeat=True))

        while symbol.flat_expression[-1]!='>':    
            symbol = self.library[''].predict()
            data.append(symbol.expand(sampled_repeat=True))
        
        return flatten(data)

    def generate_dataset(self, size=100):
        dataset = []
        for _ in range(size):
            data = self.generate()
            dataset.append(data)
        return dataset

    def get_dataset_ll(self, dataset, samples=3):

        lls = []
        for data in dataset:
            lls.append(self.get_ll(data, samples=samples))

        return np.sum(lls)

    def get_ll(self, data, samples=3):

        ll_estimate = 0
        for _ in range(samples):

            symbol, memoized_children = self.compress(data)
            l = symbol.probability(data, 0)
            ll = np.log2(l)
            ll_estimate += ll

            for child in memoized_children:
                self.remove(child)

        return ll_estimate/samples

    def mdl(self, dataset):
        
        mdl = - self.get_dataset_ll(dataset) + self.get_entropy()
        # n_char = len(flatten(dataset))
        # return mdl/n_char
        return mdl

    def write_to_txt(self, file_path):

        with open(file_path, 'w') as f:
            for key, node in self.library.items():
                # Extract the keys (expressions) of the children
                children_keys = [child.expression for child in node.children]
                # Write the key and its children to the file
                f.write(f"{key}: {children_keys}\n")

    def plot(self, save_name=None):

        # TODO: Decide if we want to plot the entire library or only the important library
        # important_library = self.get_important_library(threshold=.95)
        important_library = self.library

        # Sort library alphabetically
        important_library = dict(sorted(important_library.items(), key=lambda item: item[0]))
        nonterminals = [node for node in important_library.values() if node.type=='nonterminal']

        edges = []
        for parent_node_expression, parent_node in important_library.items():
            for node_expression, node in important_library.items():
                if node in parent_node.children:
                    edges.append((parent_node_expression, node_expression))

        # Create a directed graph
        G = nx.DiGraph()

        for edge in edges:
            # G.add_edge(edge[0], edge[1], weight=edge[2])
            G.add_edge(edge[0], edge[1])

        # weights = [G[u][v]['weight'] for u, v in G.edges()]
        # normalized_weights = [w / max(weights) * 10 for w in weights]

        # pos = graphviz_layout(G, prog='dot')
        pos = tree_layout(G, root='')  # Root of tree is the empty string

        # only color nonterminals; terminals will stay black
        cmap = plt.cm.get_cmap('tab20', len(nonterminals))
        colors = [cmap(i) for i in range(len(nonterminals))]
        colormap = {node.expression: color for node, color in zip(nonterminals, colors)}

        # Library order is max node order
        library_order = max([node.order for node in important_library.values()])
        _, ax = plt.subplots(figsize=(20, library_order))
       
        # nx.draw_networkx_edges(G, pos, width=normalized_weights, ax=ax)  # Set edge widths based on weights
        nx.draw_networkx_edges(G, pos, ax=ax, width=3)
        nx.draw_networkx_labels(G, pos, font_color='black', font_size=20, ax=ax)

        # Draw nodes and edges separately to customize their properties
        # nx.draw_networkx_nodes(G, pos, node_color='white')
        rect_width = 0.02
        rect_height = 0.1
        for node, (x, y) in pos.items():
            color = colormap.get(node, 'white')  # Default color if node color not specified
            scaling_factor = max(1, len(node))
            width = rect_width*scaling_factor  # increase minimum width so that boxes for terminals are not too narrow
            height = rect_height

            rectangle = mpatches.FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                            boxstyle="square,pad=0",
                                            ec='black', fill=True, facecolor=color, lw=3, transform=ax.transData)
            ax.add_patch(rectangle)

        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=100)
            plt.close('all')
        else:
            plt.show()

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
        # Compute the total length of all child expressions
        total_length = sum(len(neighbor) for neighbor in neighbors)

        # Adjust the total width based on the total length of the children
        adjusted_width = max(width, total_length * 0.001 * len(neighbors))

        # Divide the horizontal space proportionally based on the length of each child's expression
        next_x = xcenter - adjusted_width / 2
        for neighbor in neighbors:
            # Proportional width for this child
            child_width = adjusted_width * (len(neighbor) / total_length)
            next_x += child_width / 2  # Center the child within its allocated space
            pos = tree_layout(G, neighbor, width=child_width, vert_gap=vert_gap,
                              vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos, parent=root)
            next_x += child_width / 2  # Move to the next child's position

    return pos