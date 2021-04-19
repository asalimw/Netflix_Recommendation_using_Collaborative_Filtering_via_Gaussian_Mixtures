# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import common
import kmeans
import matplotlib.pyplot as plt

# -----------------------------------
# K-Means Algorithm
# The K-means algorithm will only care about the means,
# however, and returns a mixture that is retrofitted based on the K-means solution
# -----------------------------------

# https://www.kaggle.com/charel/learn-by-example-expectation-maximization
# https://www.youtube.com/watch?v=JNlEIEwe-Cg - informal explanation of GMM
# https://vannevar.ece.uw.edu/techsite/papers/documents/UWEETR-2010-0002.pdf
# https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
# https://github.com/llSourcell/Gaussian_Mixture_Models/blob/master/intro_to_gmm_%26_em.ipynb

X = np.loadtxt('toy_data.txt') # Read a 2D toy dataset
K = np.array([1,2,3,4]) # Initialise the no. of clusters
seeds = np.array([0,1,2,3,4]) # random seed used to randomly initialize the parameters.
# print (X)
a = X[:,0]
b = X[:,1]
# print("")
# print(a)
# print("")
# print(b)
plt.scatter(a, b)
plt.show()

for k in K:
    mixtures = []
    posts = []
    costs = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        # initialize mixture model with random points
        # init(X,K) returns a K-component mixture model with means, variances and mixing proportions.
        mixture, post = common.init(X, K=k, seed=seed)
        # print(mixture)
        # print(post)

        # run k-means
        mixture, post, cost = kmeans.run(X, mixture=mixture, post=post)
        # print(mixture)
        # print(post)
        # print(cost)

        mixtures.append(mixture)
        posts.append(post)
        costs[i] = cost
        # print(mixture)
        # print(post)
        # print(k, seed, costs)

    best_seed = np.argmin(costs)
    cost = costs[best_seed]
    mixture = mixtures[best_seed]
    post = posts[best_seed]

    # print(f'K={k}', f'Best seed: {best_seed}', f'Cost: {cost}')

    # common.plot(X, mixture, post, title=f"K-Means, K={k}")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
