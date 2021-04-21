"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


# Helpful tips for EM Algorithm
# https://www.kaggle.com/charel/learn-by-example-expectation-maximization
# https://www.youtube.com/watch?v=JNlEIEwe-Cg - informal explanation of GMM
# https://github.com/llSourcell/Gaussian_Mixture_Models/blob/master/intro_to_gmm_%26_em.ipynb

# X: an (n,d) Numpy array of n data points, each with d features
# K: number of mixture components
# mu: (K,d) Numpy array where the jth row is the mean vector μ(j)
# p: (K,) Numpy array of mixing proportions πj, j=1,…,K
# var: (K,) Numpy array of variances σ2j, j=1,…,K

# The convergence criteria is that the improvement in the log-likelihood is less than or equal to 10^−6
# multiplied by the absolute value of the new log-likelihood. In slightly more algebraic notation:
# (new-log-likelihood) − (old-log-likelihood) ≤ 10^−6⋅ |new log-likelihood|

# The code will output updated versions of a GaussianMixture (with means mu, variances var and mixing proportions p)
# as well as an (n,K) Numpy array post, where post[i,j] is the posterior probability p(j|x(i)),
# and LL which is the log-likelihood of the weighted dataset.
# EM should monotonically increase the log-likelihood of the data.
# Initialize and run the EM algorithm on the toy dataset. Check that the LL values that the algorithm returns
# after each run are indeed always monotonically increasing (non-decreasing).

# Using K=3 and a seed of 0, on the toy dataset should get a log likelihood of -1388.0818 after first iteration.


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # https://stephens999.github.io/fiveMinuteStats/intro_to_em.html

    mu = mixture.mu  # (K,d) Numpy array where the jth row is the mean vector μ(j)
    var = mixture.var  # (K,) Numpy array of variances σ2j, j=1,…,K
    p = mixture.p  # (K,) Numpy array of mixing proportions πj, j=1,…,K
    n, d = X.shape  # Shape of data X with d dimensions
    K = mu.shape[0]

    # Compute weights
    log_like = np.zeros((n, K)) # Initialise Log Likelihood as 0

    # Calculate the posterior probability of each data point i belonging to Gaussian component j
    # Posterior Prob = (Prior * Likelihood)/Sum(Likelihood Gaussian components)
    # https://medium.com/analytics-vidhya/expectation-maximization-algorithm-step-by-step-30157192de9f
    # https://cse.iitrpr.ac.in/mukesh/CS623-1920/L5-6-7-Anomaly-GMM.pdf

    for i in range(n):
        for j in range(K):
            # Multivariate Gaussian, N(x; mu, var*I)
            # With Bayes Rule - Check wikipedia for the formula
            # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

            # Using Multivariate Density - Gaussian Dist (mahalanobis distance) formula
            sigma = var[j] * np.identity(d) # Covariance
            g_numerator = (-1 / 2) * ((X[i] - mu[j]).T.dot(np.linalg.inv(sigma))).dot((X[i] - mu[j]))
            g_denominator = (((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2)))
            gaussian = np.exp(g_numerator) / g_denominator

            log_like[i, j] = p[j] * gaussian

    # posterior probability, P(j|i)
    soft_counts = log_like / log_like.sum(axis=1, keepdims=True)      #post in common.py
    # print(soft_counts)

    Log_likelihood = np.log(log_like.sum(axis=1)).sum()

    return soft_counts, Log_likelihood

    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

    n, d = X.shape
    K = post.shape[1]

    n_hat = np.sum(post, axis=0)    # Adding up the posterior probability by column
    p_hat = n_hat/n     # Weight for the mixture component

    mu_hat = (post.T @ X)/n_hat.reshape(K,1)    # Compute the mean of the cluster

    # Compute the variance
    norm = np.linalg.norm(X[:, None] - mu_hat, ord=2, axis=2) ** 2
    var_hat = np.sum(post * norm, axis=0) / (n_hat * d)

    # Return optimal mean, variance and weight
    return GaussianMixture(mu_hat, var_hat, p_hat)

    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # In the E-step, the algorithm tries to guess the value of z(i) based on the parameters,
    # while in the M-step, the algorithm updates the value of the model parameters based on the guess of z^(i)
    # of the E-step. These two steps are repeated until convergence is reached.
    #
    # Repeat until convergence:

    old_likelihood = None
    new_likelihood = None
    
    while (old_likelihood is None or new_likelihood - old_likelihood >= 1e-6 * np.abs(new_likelihood)):
        old_likelihood = new_likelihood
        post, new_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_likelihood
    raise NotImplementedError
