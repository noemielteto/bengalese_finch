from utils import *
from nonstationary_markov_model import *

###############################################################################
########################### SANDBOX OF THE SANDBOX ############################
###############################################################################

# E = ['n','a','b','l','e','d','g','f']
# I = [1, 0, 0, 0, 0, 0, 0, 0]
# S = np.array([
#                 [0.8, 0.15,   0,   0,   0,   0,   0, 0.05],
#                 [0,   0,   1,   0,   0,   0,   0, 0  ],
#                 [0,   0,   0,   1,   0,   0,   0, 0  ],
#                 [0,   0,   0,   0,   1,   0,   0, 0  ],
#                 [0.3, 0,   0,   0,   0,   0.7, 0, 0  ],
#                 [0,   0,   0,   0,   0,   0,   1, 0  ],
#                 [0,   0,   0,   0,   0,   0,   0, 1  ],
#                 [0.4, 0.6, 0,   0,   0,   0,   0, 0  ]
#                 ])
#
# decay_rate = 0.2
# restore_rate = 0.001
# # s_threshold = 0.1
#
# bout = 'nnnnnnnnnabledgfabledgnnnnnnabledgfabledgfabledgfabledgfabledgnnnnabledgfabledgfnnfabledg'

data = get_data()
bouts = data['wh09pk88'][:100]

E = list(np.unique(flatten(bouts)))

init_unigram_marginals, unigram_marginals, bigram_marginals = compute_stationary_distr(bouts, E)

I = init_unigram_marginals
S = bigram_marginals

s_thresholds = np.linspace(0.05, 0.2, 10)
decay_rates = np.linspace(0.1, 0.5, 10)
restore_rates = np.linspace(0.0001, 0.1, 10)
d = np.zeros((len(decay_rates), len(restore_rates), len(s_thresholds)))
for i, s_threshold in enumerate(s_thresholds):
    print(i)
    for j, decay_rate_value in enumerate(decay_rates):
        print(j)
        for k, restore_rate_value in enumerate(restore_rates):
            # print(k)
            theta = (decay_rate_value, restore_rate_value, s_threshold)
            d[i][j][k] = NSMM_edit_distance_allbouts_knownmatrix(theta, E, I, S, bouts)

minval_i, minval_j, minval_k = np.unravel_index(d[:5].argmin(), d.shape)  # TODO not the right pos on heatmap
f, ax = plt.subplots(1, len(s_thresholds), figsize=(2*len(s_thresholds), 2))
for i, s_threshold in enumerate(s_thresholds):
    sns.heatmap(d[i],vmin=0,ax=ax[i])
    ax[i].set_title(np.round(s_threshold,2))
    ax[i].invert_yaxis()
    if i==minval_i:
        minval_x, minval_y = minval_k, minval_j  # matrix index to figure index
        ax[i].scatter(minval_x, minval_y, marker='X', s=100, c='r')
    plt.ylabel('decay rate')
    plt.xlabel('restore rate')
plt.show()

opt_decay_rate, opt_restore_rate, opt_s_threshold = decay_rates[minval_i], restore_rates[minval_j], s_thresholds[minval_k]
print((opt_decay_rate, opt_restore_rate, opt_s_threshold))
opt_predicted_bout = NSMM_generative_knownmatrix((opt_decay_rate, opt_restore_rate, opt_s_threshold), E, I, S)
print(opt_predicted_bout)

g = NSMM_generative_knownmatrix((opt_decay_rate, opt_restore_rate, opt_s_threshold), E, I, S)
# g = NSMM_generative_knownmatrix((.3, .01, .1), E, I, S)
sum([edit_distance(g,bout) for bout in bouts])

### Optimize w gradients
#
# decay_rate_bounds = (0.01, 0.2)
# restore_rate_bounds = (0.00001, 0.1)
# s_threshold_bounds = (0.05, 0.15)
# x0 = [np.random.uniform(decay_rate_bounds[0], decay_rate_bounds[1]), np.random.uniform(restore_rate_bounds[0], restore_rate_bounds[1]), np.random.uniform(s_threshold_bounds[0], s_threshold_bounds[1])]
# print(x0)
# # x0 = [0.1, 0.01,0.1]
# res = minimize(NSMM_edit_distance_allbouts_knownmatrix, x0, args=(E, I, S, bouts), method='L-BFGS-B', bounds=(decay_rate_bounds, restore_rate_bounds, s_threshold_bounds), tol=1e-6)
# print(res)
# print(NSMM_edit_distance(res.x, E, I, S, bout))
#
#
#
# prob_bouds = (0,1)
# decay_rate_bounds = (0.01, 0.2)
# restore_rate_bounds = (0.00001, 0.1)
# s_threshold_bounds = (0.05, 0.15)
# # x0 = list(np.random.uniform(0,1,len(E)+len(E)**2)) + [np.random.uniform(decay_rate_bounds[0], decay_rate_bounds[1]), np.random.uniform(restore_rate_bounds[0], restore_rate_bounds[1]), np.random.uniform(s_threshold_bounds[0], s_threshold_bounds[1])]
#
#
# data = get_data()
# bouts = data['wh09pk88'][:10]
# init_unigram_marginals, unigram_marginals, bigram_marginals = compute_stationary_distr(bouts, E)
#
# best_sum_edit_distance = 10**10
# for repeat in range(10):
#
#     print('-------------------------------------------------------------------')
#     print(repeat)
#
#     noisy_init_unigram_marginals = init_unigram_marginals+np.random.normal(0,0.01,init_unigram_marginals.shape)
#     noisy_init_unigram_marginals = np.where(noisy_init_unigram_marginals<0, 0, noisy_init_unigram_marginals)
#     noisy_init_unigram_marginals = np.where(noisy_init_unigram_marginals>1, 1, noisy_init_unigram_marginals)
#     noisy_init_unigram_marginals = normalize_matrix(noisy_init_unigram_marginals)
#
#     noisy_bigram_marginals = bigram_marginals+np.random.normal(0,0.01,bigram_marginals.shape)
#     noisy_bigram_marginals = np.where(noisy_bigram_marginals<0, 0, noisy_bigram_marginals)
#     noisy_bigram_marginals = np.where(noisy_bigram_marginals>1, 1, noisy_bigram_marginals)
#     noisy_bigram_marginals = normalize_matrix(noisy_bigram_marginals)
#
#     I_estimate = list(noisy_init_unigram_marginals)
#     S_estimate = list(noisy_bigram_marginals.ravel())
#
#     x0 = I_estimate + S_estimate + [np.random.uniform(decay_rate_bounds[0], decay_rate_bounds[1]), np.random.uniform(restore_rate_bounds[0], restore_rate_bounds[1]), np.random.uniform(s_threshold_bounds[0], s_threshold_bounds[1])]
#
#     print(x0)
#     res = minimize(NSMM_edit_distance_allbouts, x0, args=(E, bouts), method='L-BFGS-B', bounds=[prob_bouds]*(len(E)+len(E)**2)+[decay_rate_bounds, restore_rate_bounds, s_threshold_bounds], tol=1e-10)
#     print(res)
#     sum_edit_distance = NSMM_edit_distance_allbouts(res.x, E, bouts)
#     print(sum_edit_distance)
#     if sum_edit_distance<best_sum_edit_distance:
#         print('improved')
#         best_sum_edit_distance = sum_edit_distance
#         best_theta = res.x
