from utils import *
from scipy.optimize import minimize

def get_E_I_S(bouts):
    E = list(np.unique(flatten(bouts)))
    init_unigram_marginals, unigram_marginals, bigram_marginals = compute_stationary_distr(bouts, E)
    I = init_unigram_marginals
    S = bigram_marginals
    return E,I,S

def NSMM_generative_knownmatrix(theta, E, I, S, probabilistic=False, draw_matrices=False):

    decay_rate, restore_rate, s_threshold = theta[-3:]
    S_bout = copy.deepcopy(S)

    predicted_bout  = []
    # next_e = np.random.choice(a=E, p=I)  # probabilistic start
    next_e = E[np.argmax(I)]
    s = 10
    i = 0

    while s>s_threshold and len(predicted_bout)<500:

        current_e = next_e
        # print(current_e)
        predicted_bout.append(current_e)

        T = normalize_matrix(S_bout)
        if draw_matrices:
            sns.heatmap(np.round(T,2),vmin=0,vmax=1,annot=True)
            plt.savefig('T_'+str(i)+'.png')
            plt.close('all')

        if probabilistic:
            next_e = np.random.choice(a=E, p=T[E.index(current_e)])  # sample transition probabilistically
        else:
            next_e = E[np.argmax(T[E.index(current_e)])]  # winner takes all -- pick highest transition prob

        s = S_bout[E.index(current_e)][E.index(next_e)]

        # decay
        S_bout[E.index(current_e)][E.index(next_e)] -= S_bout[E.index(current_e)][E.index(next_e)] * decay_rate

        # restore
        diff_from_marginal = copy.deepcopy(S) - S_bout
        S_bout += diff_from_marginal * restore_rate

        i+=1

    return predicted_bout

def NSMM_generative(theta, E):

    decay_rate, restore_rate, s_threshold = theta[-3:]
    I = normalize_matrix(theta[:len(E)])
    S = np.array(theta[len(E):len(E) + len(E)**2])
    # TODO stack array?

    predicted_bout  = []
    # next_e = np.random.choice(a=E, p=I)  # probabilistic start
    next_e = E[np.argmax(I)]
    s = 10

    while s>s_threshold and len(predicted_bout)<1000:

        current_e = next_e
        # print(current_e)
        predicted_bout.append(current_e)

        T = normalize_matrix(S)
        # next_e = np.random.choice(a=S, p=T[S.index(current_e)])  # sample transition probabilistically
        next_e = E[np.argmax(T[E.index(current_e)])]  # winner takes all -- pick highest transition prob

        s = S[E.index(current_e)][E.index(next_e)]

        # decay
        S[E.index(current_e)][E.index(next_e)] -= S[E.index(current_e)][E.index(next_e)] * decay_rate

        # restore
        S += S * restore_rate

    return predicted_bout

def NSMM_edit_distance(theta, E, bout):

    predicted_bout = NSMM_generative(theta, E)
    return edit_distance(predicted_bout, bout)

def NSMM_edit_distance_allbouts(theta, E, bouts, I=None, S=None):

    predicted_bout = NSMM_generative(theta=theta, E=E, I=I, S=S)
    sum_edit_distance = 0
    for bout_i, bout in enumerate(bouts):
        # print(bout_i)
        sum_edit_distance += edit_distance(predicted_bout, bout)

    return sum_edit_distance

def NSMM_edit_distance_allbouts_knownmatrix(theta, E, I, S, bouts, probabilistic=False, return_predicted_bout=False):

    predicted_bout = NSMM_generative_knownmatrix(theta=theta, E=E, I=I, S=S, probabilistic=probabilistic)
    # print(f'params: {theta}')
    # print(predicted_bout)
    # print('--------------------------------------------------------------')
    sum_edit_distance = 0
    for bout_i, bout in enumerate(bouts):
        # print(bout_i)
        sum_edit_distance += edit_distance(predicted_bout, bout)

    if return_predicted_bout:
        return predicted_bout, sum_edit_distance
    else:
        return sum_edit_distance

def NSMM_single_likelihood(T, E, previous_e, current_e, binary=False):

    if not binary:
        # transition likelihood
        return T[E.index(previous_e), E.index(current_e)]

    else:
        # binary likelihood
        transprobs = T[E.index(previous_e)]
        return 1 if E[transprobs.argmax()] == current_e else 0.1**(10)

def NSMM_recognition(theta, E, I, S, bout, binary=False):

    if len(bout)==0:
        return np.array([])

    decay_rate, restore_rate, s_threshold = theta
    S_bout = copy.deepcopy(S)  # important to only update the copy of S!

    current_e = bout[0]
    likelihoods = []
    if binary:
        likelihood = 1 if E[np.argmax(I)] == current_e else 0.1**(10)
    else:
        likelihood = I[E.index(current_e)]
    likelihoods.append(likelihood)

    # while s>s_threshold:  # would be used for generative case
    for i in range(1, len(bout)):

        previous_e, current_e = bout[i-1], bout[i]

        # decay
        S_bout[E.index(previous_e), E.index(current_e)] -= S_bout[E.index(previous_e), E.index(current_e)] * decay_rate

        # restore
        diff_from_marginal = copy.deepcopy(S) - S_bout
        S_bout += diff_from_marginal * restore_rate

        # s = S[E.index(previous_e), E.index(current_e)]

        T = normalize_matrix(S_bout)
        likelihoods.append(NSMM_single_likelihood(T, E, previous_e, current_e, binary))

    # print(list(zip(bout,likelihoods)))

    likelihoods = np.array(likelihoods)
    likelihoods = np.where(likelihoods==0, 0.1**(10), likelihoods)  # mask 0s

    # return sum(-np.log(likelihoods))
    return likelihoods

def NSMM_recognition_allbouts(theta, E, I, S, bouts, binary=False):
    likelihoods = np.array([])
    for bout in bouts:
        likelihoods = np.concatenate((likelihoods, NSMM_recognition(theta, E, I, S, bout, binary=binary)))
    return likelihoods
