from utils import *
from nonstationary_markov_model import *
from HCRP_LM.ddHCRP_LM import HCRP_LM

data = get_data()
subjects = data.keys()
d = pd.read_csv('gridsearch_values_2.csv')

palette = ['#D81B60', '#1E88E5', '#FFC107', '#004D40', '#A2A43C', '#26857C', '#E26570', '#991771', '#987CCA', '#11E1FF', '#E7590A', '#447F7E']

subject = 'wh09pk88'

bouts = data[subject]
train_bouts = bouts[:int(len(bouts)*.8)]
test_bouts = bouts[int(len(bouts)*.8):]
subd = d[d['subject']==subject]
best_theta = subd.iloc[subd.edit_distance.argmin()][['decay_rate', 'restore_rate', 's_threshold']].tolist()
E, I, S = get_E_I_S(bouts)

############################### binary recognition test ###############################

HCRP_model = HCRP_LM([1]*2)
HCRP_model.fit(train_bouts, frozen=True)

NSMM_likelihoods = NSMM_recognition_allbouts(best_theta, E, I, S, test_bouts, binary=True)
HCRP_model.predict(test_bouts)
# HCRP_likelihoods = HCRP_model.choice_probs
HCRP_likelihoods = [1 if syl == HCRP_model.dishes[np.argmax(HCRP_model.predictive_distr[i])] else 0 for i,syl in enumerate(flatten(test_bouts))]
# print(sum(-np.log(NSMM_likelihoods)))
# print(sum(-np.log(HCRP_likelihoods)))

print(sum(NSMM_likelihoods))
print(sum(HCRP_likelihoods))

# print(list(zip(bout,np.round(HCRP_likelihoods,2), np.round(NSMM_likelihoods,2))))

len_firstbout = len(test_bouts[0])
plt.scatter(range(len_firstbout), HCRP_likelihoods[:len_firstbout])
plt.scatter(range(len_firstbout), NSMM_likelihoods[:len_firstbout]+0.01)
plt.xticks(range(len_firstbout),test_bouts[0])
plt.show()

########################### recognition test ############################

NSMM_likelihoods = NSMM_recognition_allbouts(best_theta, E, I, S, test_bouts)
HCRP_likelihoods = HCRP_model.choice_probs
print(sum(-np.log(NSMM_likelihoods)))
print(sum(-np.log(HCRP_likelihoods)))

# print(list(zip(bout,np.round(HCRP_likelihoods,2), np.round(NSMM_likelihoods,2))))

len_firstbout = len(test_bouts[0])
plt.scatter(range(len_firstbout), HCRP_likelihoods[:len_firstbout])
plt.scatter(range(len_firstbout), NSMM_likelihoods[:len_firstbout])
plt.xticks(range(len_firstbout),test_bouts[0])
plt.show()

############################### openloop generative test #######################

# use start/stop tokens in order to be able to generate openloop
train_bouts_start_stop = [['START']+bout+['STOP'] for bout in train_bouts]
HCRP_model = HCRP_LM([1]*25)
HCRP_model.fit(train_bouts_start_stop, frozen=True)

sum_HCRP_match_proportion = 0
sum_NSMM_match_proportion = 0
n_repeats = 10
for rep in range(n_repeats):

    print(rep)

    ### NSMM ###

    NSMM_predicted_bout, NSMM_edit_distance_allbouts = NSMM_edit_distance_allbouts_knownmatrix(best_theta, E, I, S, test_bouts, probabilistic=False, return_predicted_bout=True)
    longer_length = max(len(flatten(test_bouts)), len(NSMM_predicted_bout*len(test_bouts)))  # TODO will need to think deeper about this -- probably should use non-symetric edit distance!
    NSMM_match_proportion =  1 - (NSMM_edit_distance_allbouts/longer_length)
    sum_NSMM_match_proportion += NSMM_match_proportion

    ### HCRP ###

    HCRP_predicted_bout = []
    syl = 'START'
    while syl!='STOP':
        HCRP_predicted_bout.append(syl)
        syl = HCRP_model.predict_next_word(t=0,u=HCRP_predicted_bout)
    HCRP_predicted_bout = HCRP_predicted_bout[1:]  # drop start token
    HCRP_edit_distance_allbouts = sum([edit_distance(HCRP_predicted_bout, test_bout) for test_bout in test_bouts])
    longer_length = max(len(flatten(test_bouts)), len(HCRP_predicted_bout*len(test_bouts)))  # TODO will need to think deeper about this -- probably should use non-symetric edit distance!
    HCRP_match_proportion =  1 - (HCRP_edit_distance_allbouts/longer_length)
    sum_HCRP_match_proportion += HCRP_match_proportion

    print(''.join(HCRP_predicted_bout))
    print(''.join(NSMM_predicted_bout))
    # print(np.mean([len(bout) for bout in test_bouts]))
    print('----------------------')

print(sum_HCRP_match_proportion/n_repeats)
print(sum_NSMM_match_proportion/n_repeats)
