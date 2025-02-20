from PST import *
from func import *

pkl = 'pst_result.pkl'

depth = 12
data = get_data()

for subject in data.keys():

    X = data[subject][:20]

    # Training data is last 90% of the songs
    X_train = X[int(len(X)/10):]
    X_test  = X[:int(len(X)/10)]

    tree = gen_tree(X=X_train, L=depth)
    # draw_pst(tree)
    tree.predict(X=X_test)
    PST_likelihoods = tree.likelihoods

    m = HCRP_LM(strength=[1]*(depth+1))
    m.fit(X_train)
    m.predict(X_test)
    HCRP_likelihoods = m.likelihoods

    print(np.mean(PST_likelihoods))
    print(np.mean(HCRP_likelihoods))
    print('-----------------')

    # np.mean(HCRP_likelihoods)
    # np.mean(likelihoods)
    # plt.scatter(HCRP_likelihoods, likelihoods, alpha=0.5)
    # plt.show()
    # plt.hist(HCRP_likelihoods, alpha=0.5)
    # plt.hist(likelihoods, alpha=0.5)
    # plt.show()
