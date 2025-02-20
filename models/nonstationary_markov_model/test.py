from utils import *
from nonstationary_markov_model import *


data = get_data()
subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']
d = pd.read_csv('gridsearch_values_2.csv')

palette = ['#D81B60', '#1E88E5', '#FFC107', '#004D40', '#A2A43C', '#26857C', '#E26570', '#991771', '#987CCA', '#11E1FF', '#E7590A', '#447F7E']

match_proportions = []
between_bouts_match_proportions = []

for subject in subjects:
# for subject in ['wh09pk88']:
    bouts = data[subject]

    train_bouts = bouts[:int(len(bouts)*.8)]
    E = list(np.unique(flatten(train_bouts)))
    init_unigram_marginals, unigram_marginals, bigram_marginals = compute_stationary_distr(train_bouts, E)
    I = init_unigram_marginals
    S = bigram_marginals

    test_bouts = bouts[int(len(bouts)*.8):]
    subd = d[d['subject']==subject]
    best_theta = subd.iloc[subd.edit_distance.argmin()][['decay_rate', 'restore_rate', 's_threshold']].tolist()

    edit_distance_allbouts = NSMM_edit_distance_allbouts_knownmatrix(best_theta, E, I, S, test_bouts)
    predicted_bout = NSMM_generative_knownmatrix(best_theta, E, I, S)
    longer_length = max(len(flatten(test_bouts)), len(predicted_bout*len(test_bouts)))  # TODO will need to think deeper about this -- probably should use non-symetric edit distance!
    match_proportion =  1 - (edit_distance_allbouts/longer_length)
    match_proportions.append(match_proportion)

    print(subject)
    print(best_theta)
    print(predicted_bout)
    print(match_proportion)

    subject_between_bouts_match_proportions = []
    for bout_1 in test_bouts:
        for bout_2 in test_bouts:
            if len(bout_1)==0 or len(bout_2)==0:
                continue
            between_bout_edit_distance = edit_distance(bout_1, bout_2)
            longer_length = max(len(bout_1), len(bout_2))
            match_proportion =  1 - (between_bout_edit_distance/longer_length)
            subject_between_bouts_match_proportions.append(match_proportion)

    between_bouts_match_proportions.append(np.mean(subject_between_bouts_match_proportions))
    #
    # predicted_bout_stationary = NSMM_generative_knownmatrix((0,0,0), E, I, S)[:len(predicted_bout)]
    # edit_distance_allbouts_stationary = sum([edit_distance(bout, predicted_bout_stationary) for bout in test_bouts])
    # match_proportion_stationary =  1 - (edit_distance_allbouts_stationary/longer_length)
    #
    # print(predicted_bout_stationary)
    # print(match_proportion_stationary)
    print('-------------------------------------------------------------------')

    # f,ax=plt.subplots(1,1,figsize=(13,3))
    # for i, bout in enumerate(bouts[:10]):
    #     bout_colorseq = [palette[E.index(syl)] for syl in bout]
    #     ax.scatter(range(len(bout)), [i]*len(bout), c=bout_colorseq)
    #     for j, s in enumerate(bout):
    #         ax.annotate(text=s,xy=(j-0.5,i))
    #
    # predicted_bout_colorseq = [palette[E.index(syl)] for syl in predicted_bout]
    # ax.scatter(range(len(predicted_bout)), [-1]*len(predicted_bout), c=predicted_bout_colorseq)
    # for j, s in enumerate(predicted_bout):
    #     ax.annotate(text=s,xy=(j-0.5,-1))
    # ax.axhline(y=-0.5,c='k')
    # ax.set_title(subject)
    # plt.tight_layout()
    # plt.savefig(subject+'_2.png')
    # plt.close('all')

#
plt.plot(match_proportions, label='match proportion with prediction')
plt.plot(between_bouts_match_proportions, label='between bouts match proportion')
plt.ylim(0,1)
plt.xlabel('subject')
plt.ylabel('match proportion')
plt.legend()
plt.tight_layout()
plt.show()
#
# ################################################################################
#
# from HCRP_LM.ddHCRP_LM import HCRP_LM
#
# LLs_predicted_bout = []
# LLs_real_bout = []
# depths = [1, 2, 3, 4, 5, 6, 7, 8]
# for depth in depths:
#     print(depth)
#     m_1 = HCRP_LM([1]*depth)
#     m_1.fit([predicted_bout]*100, frozen=True)
#     LLs_predicted_bout.append(-m_1.negLL())
#
#     m_2 = HCRP_LM([1]*depth)
#     m_2.fit(bouts[:100], frozen=True)
#     LLs_real_bout.append(-m_2.negLL())
#
# plt.plot(LLs_predicted_bout[1:], label='generated')
# plt.plot(LLs_real_bout[1:], label='real')
# plt.xticks(range(len(depths[1:])), [d-1 for d in depths[1:]])
# plt.xlabel('context depth used for prediction')
# plt.ylabel('log likelihood')
# plt.legend()
# plt.tight_layout()
# plt.show()
