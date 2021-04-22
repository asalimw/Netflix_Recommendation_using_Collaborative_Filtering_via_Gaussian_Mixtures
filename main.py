# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import common
import kmeans
# import matplotlib.pyplot as plt
import naive_em
print("import done")

# -----------------------------------
# K-Means Algorithm
# The K-means algorithm will only care about the means,
# however, and returns a mixture that is retrofitted based on the K-means solution
# -----------------------------------

# https://www.kaggle.com/charel/learn-by-example-expectation-maximization
# https://www.youtube.com/watch?v=JNlEIEwe-Cg - informal explanation of GMM
# https://vannevar.ece.uw.edu/techsite/papers/documents/UWEETR-2010-0002.pdf
# https://github.com/llSourcell/Gaussian_Mixture_Models/blob/master/intro_to_gmm_%26_em.ipynb

X = np.loadtxt('toy_data.txt') # Read a 2D toy dataset
K = np.array([1,2,3,4]) # Initialise the no. of clusters# https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

seeds = np.array([0,1,2,3,4]) # random seed used to randomly initialize the parameters.

# BIC score of cluster
bic = np.zeros(len(K))

# print (X)
# print(len(X))
a = X[:,0]
b = X[:,1]
# print("")
# print(a)
# print("")
# print(b)
# plt.scatter(a, b)
# plt.show()

for j, k in enumerate(K):
    #Initialise empty vector for K-means
    mixtures = []
    posts = []
    costs = np.empty(len(seeds))

    # Initialise empty vector for EM Algorithm
    mixtures_em = []
    posts_em = []
    costs_em = np.empty(len(seeds))
    logloss = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        # initialize mixture model with random points
        # init(X,K) returns a K-component mixture model with means, variances and mixing proportions.
        mixture, post = common.init(X, K=k, seed=seed)
        mixture_em, post_em = common.init(X, K=k, seed=seed)  # For EM algorithm initialisation

        # run k-means function
        mixture, post, cost = kmeans.run(X, mixture=mixture, post=post)

        # run EM Algo function
        mixture_em, post_em, ll = naive_em.run(X, mixture=mixture_em, post=post_em)

        # Update k-means values
        mixtures.append(mixture)
        posts.append(post)
        costs[i] = cost
        # print(k, seed, costs)

        # Update EM values
        mixtures_em.append(mixture_em)
        posts_em.append(post_em)
        logloss[i] = ll
        # print(k, seed, costs_em)


    # Finding the best/min cost of k-means
    best_seed = np.argmin(costs)    # set the best seed by finding min value
    cost = costs[best_seed]         # update cost with the best seed index
    mixture = mixtures[best_seed]   # update mixture with best seed index
    post = posts[best_seed]         # update post/soft assignment with best seed index


    # Finding the best/min cost of EM Algorithm
    best_seed_em = np.argmax(logloss)      # set the best seed by finding min value
    logloss = logloss[best_seed_em]        # update cost with the best seed index
    mixture_em = mixtures_em[best_seed_em]  # update mixture with best seed index
    post_em = posts_em[best_seed_em]        # update post/soft assignment with best seed index

    # Print output and graph of k-means min cost
    print("k-means method:", f'K={k}', f'Best seed: {best_seed}', f'Cost: {cost}')
    common.plot(X, mixture, post, title=f"K-Means, K={k}")

    # Print output and graph of EM algorithm min cost
    print("EM method:", f'K={k}', f'Best seed: {best_seed_em}', f'Cost: {logloss}')
    common.plot(X, mixture_em, post_em, title=f"EM Algorithm, K={k}")

    # BIC score for EM
    current_bic = common.bic(X, mixture_em, logloss)
    bic[j] = current_bic
    # print(f'K={k}', f'Best seed={best_seed_em}', f'logloss={logloss}', f'BIC={current_bic}')

# In a situation where we wish to select models, we want a model with the the highest BIC.
best_K_ix = np.argmax(bic)
best_K = K[best_K_ix]
best_bic = bic[best_K_ix]
print(f"Best K={best_K}", f"BIC={best_bic}")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
