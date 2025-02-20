# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib3.DataAndObjects import FunctionData

# def make_data(size=1, alpha=0.99):
#     return [FunctionData(input=[],   output=['NNNABLEDGFABLEDGF'], alpha=alpha),
#             FunctionData(input=[],   output=['NNNNNNABLEDGF'], alpha=alpha),
#             FunctionData(input=[],   output=['NNNNNNABLEDGFABLE'], alpha=alpha),
#             FunctionData(input=[],   output=['NNNNNNABLEDGFABLEDGF'], alpha=alpha),
#             FunctionData(input=[],   output=['NNNNNNABLEDGF'], alpha=alpha)
#             ] * size

# def make_data(size=1, alpha=0.99):
#     return [FunctionData(input=['NNNABLEDGFABLEDGF'],   output=True, alpha=alpha),
#             FunctionData(input=['NNNNNNABLEDGF'],       output=True, alpha=alpha),
#             FunctionData(input=['NNNNNNABLEDGFABLE'],   output=True, alpha=alpha),
#             FunctionData(input=['NNNNNNABLEDGFABLEDGF'],output=True, alpha=alpha),
#             FunctionData(input=['NNNNNNABLEDGF'],       output=True, alpha=alpha)
#             ] * size

# def make_data(size=1, alpha=0.99):
#     return [FunctionData(input=['AB'],          output=True, alpha=alpha),
#             FunctionData(input=['ABAB'],        output=True, alpha=alpha),
#             FunctionData(input=['AAA'],         output=False, alpha=alpha),
#             FunctionData(input=['ABABAB'],      output=True, alpha=alpha),
#             FunctionData(input=['BBB'],         output=False, alpha=alpha),
#             FunctionData(input=['BABA'],         output=False, alpha=alpha),
#             ] * size

# def make_data(size=1, alpha=0.99):
#     return [FunctionData(input=[], output='NNNABLEDGFABLEDGF', alpha=alpha)
#             ] * size

def make_data(size=1, alpha=0.99):
    return [FunctionData(input=[], output='nnnnnnnnnabledgfablennnnnnnabledgfabledgfabledgfabledgfabledgnnnnabledgfabledgfnnfabledg', alpha=alpha)
            ] * size



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib3.Grammar import Grammar
from LOTlib3.Miscellaneous import q

grammar = Grammar()

# SYLS = ['N','A','B','L','E','D','G','F']
SYLS = ['n','a','b','l','e','d','g','f']
# SYLS = ['A','B']
INTS  = range(1,10)
# INTS  = range(1,4)

grammar.add_rule('START', '', ['EXPR'], 1.0)
grammar.add_rule('EXPR', '(%s+%s)', ['EXPR', 'EXPR'], 1.0)  # concat
grammar.add_rule('EXPR', '(%s*%s)', ['EXPR', 'INT'], 1.0)  # repeat

# grammar.add_rule('EXPR', '(%s+%s)', ['EXPR', 'SYL'], 1.0)  # no need?
# grammar.add_rule('EXPR', '(%s+%s)', ['SYL', 'EXPR'], 1.0)  # no need?
grammar.add_rule('EXPR', '%s', ['SYL'], 2)

for syl in SYLS:
    grammar.add_rule('SYL', q(syl), None, 1.0)
for integer in INTS:
    grammar.add_rule('INT', str(integer), None, 1.0)

example = grammar.generate()
# ('N'*6) + ( ('A'+('B'+('L'+'E'))) + ('D'+('G'+'F')) )*2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib3.FunctionNode import isFunctionNode
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib3.Eval import EvaluationException
import re
from math import log

def edit_distance(s1, s2):
    # https://stackoverflow.com/questions/2460177/edit-distance-in-python
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

class MyHypothesis(LOTHypothesis):
    """Define a special hypothesis.
    This requires overwriting compile_function to use our custom interpretation model on trees -- not just
    simple eval.
    """

    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar, **kwargs)

    # def compile_function(self):
    #     c = re.compile(str(self.value))
    #     return (lambda s: (c.match(s) is not None))

    # Instead of BinaryLikelihood, we'll have edit distance LL
    def compute_single_likelihood(self, datum):
        try:

            binary_likelihood = datum.alpha * (self(*datum.input) == datum.output) + (1.0-datum.alpha) / 2.0

            proposed = self(*datum.input)
            longer_length = max(len(proposed), len(datum.output))  # TODO will need to think deeper about this -- probably should use non-symetric edit distance!
            match_proportion =  1 - (edit_distance(proposed, datum.output)/longer_length)
            match_proportion = max(1/1000, match_proportion)  # we prevent 0 match

            # mix of binary likelihood and match proportion; mixing proportions control smoothness
            return log(0.95 * binary_likelihood + 0.05 * match_proportion)

        except RecursionDepthException as e: # we get this from recursing too deep -- catch and thus treat "ret" as None
            return -Infinity

    def __str__(self):
        return str(self.value)

    def __call__(self, *args):
        return self.fvalue

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    from LOTlib3 import break_ctrlc
    from LOTlib3.Miscellaneous import qq
    from LOTlib3.TopN import TopN
    from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler

    h0   = MyHypothesis()
    data = make_data(size=10)
    top  = TopN(N=10)
    thin = 100

    for i, h in enumerate(break_ctrlc(MetropolisHastingsSampler(h0, data))):

        top << h

        if i % thin == 0:
            print("#", i, h.posterior_score, h.prior, h.likelihood, qq(h))

        if i==30000:
            break

    for h in top:
        print(h.posterior_score, h.prior, h.likelihood, qq(h))
