"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # http://home.eng.iastate.edu/~rkumar/PUBS/matrix-comp-gm.pdf.pdf
    # https://www.researchgate.net/publication/47460816_Efficient_Matrix_Completion_with_Gaussian_Models
    # https://www.youtube.com/watch?v=sooj-_bXWgk
    # https://www.youtube.com/watch?v=lS0FVKJ4Xfg
    # https://www.youtube.com/watch?v=ZspR5PZemcs
    # https://en.wikipedia.org/wiki/LogSumExp
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
    # https://www.youtube.com/watch?v=-RVM21Voo7Q

    # Using Bayes' rule to find an updated expression for the posterior probability
    # Posterior prob P(j|u) = (Prior * Likelihood)/Sum(Likelihood Gaussian components)
    # To minimize numerical instability, you will be re-implementing the E-step in the log-domain
    # the actual output of E-step should include the non-log posterior

    # calculate the values for the log of the posterior probability l(j,u) = log(p(j|u))
    # after long calculation this is equivalent to f(u,j) - log(sum(exp(f(u,j)))

    # Generative Model for Univariate Gaussian Dist formula is:
    # sum of n * -d/2(log(2*pi*var) + sum of n * (norm(X(t) - mu)^2)/-2*var^2
    # where the sum of n is the sum of sparse matrix

    n = X.shape[0]
    mu, var, p = mixture  # Unpacking the mixture

    # Creating a matrix recording the dimension of the original matrix, each cell equals 1
    # when positive and 0 when negative
    sparse_matrix = np.where(X > 0, 1, 0)

    # Sum the sparse matrix based on column to get the dimensionality of each data point with data in it
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))   # n*d


    first_term = -d / 2 * np.log(2 * np.pi * var)   # Denominator of Gaussians distribution

    # Vectorized Exponential term of normal distribution
    exponent = np.linalg.norm((X[:, np.newaxis] - mu) * sparse_matrix[:, np.newaxis], axis=2) ** 2 / -(2 * var)

    log_p = np.log(p + 1e-16)  # log-transform the terms to prevent numerical underflow

    log_norm = first_term + exponent  # log-transform the terms to prevent numerial underflow

    f_uj = log_p + log_norm  # Redefine the log-transformed terms as a function f_uj

    log_post = f_uj - logsumexp(f_uj, axis=1).reshape((n, 1))  # The weighted soft count in log form

    # Exponential the log_post variable to get the original soft count of each data point
    origin_post = np.exp(log_post)

    # The log likelihood would then be the sum of all the logsumexp term (the total count in log form)
    log_likelihood = np.sum(logsumexp(f_uj, axis=1).reshape((n, 1)), axis=0)
    return origin_post, log_likelihood


    #--------------------------------------------
    # Calculate the f(u,j) - the MLE for Gaussian Distribution
    # f_ui[i,j] = np.log(pi[j] + 1e-16) + log_pdf[i,j]

    # log_pdf --- the log of multivariate gaussian
    # log_pdf = multivariate_normal.logpdf(X, mu, var)
    # log_p = np.log(p + 1e-16) # 1e-e16 is to avoid numerical underflow
    # f_uj = log_p + log_pdf  #
    # print(log_pdf)

    # log of the posterior probability l(j,u) = f(u,j) - log(sum(exp(f(u,j)))
    # log_post = f_uj - logsumexp(f_uj, axis=1).reshape((n, 1))  # the weighted soft count in log form

    # log_like = np.sum(logsumexp(f_uj, axis=1).reshape((n, 1)), axis=0)
    # -------------------------------------------------------------------

    # return log_post, log_likelihood

    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    # n, d = X.shape
    # _, K = post.shape
    #
    # n_hat = post.sum(axis=0)
    # p_hat = n_hat / n
    #
    # mu_hat = (1 / n_hat.reshape(K, 1)) * post.T @ X
    #
    # norm = np.power(np.linalg.norm(X[:, np.newaxis] - mu_hat, axis=2), 2)
    # summation = np.sum(post * norm, axis=0)
    #
    # var_hat = (1 / (n_hat * d)) * summation

    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = mixture.mu.copy()
    var = np.zeros(K)

    for j in range(K):
        sse, weight = 0, 0
        for l in range(d):
            mask = (X[:, l] != 0)
            n_sum = post[mask, j].sum()
            if (n_sum >= 1):
                # Updating mean
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum
            # Computing variance
            sse += ((mu[j, l] - X[mask, l]) ** 2) @ post[mask, j]
            weight += n_sum
        var[j] = sse / weight
        if var[j] < min_variance:
            var[j] = min_variance

    return GaussianMixture(mu, var, p)

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
        mixture = mstep(X, post, mixture)

    return mixture, post, new_likelihood

    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu, var, p = mixture
    post, likelihood = estep(X, mixture)
    update_indicator_matrix = np.where(X != 0, 1, 0)
    predicted_value = post @ mu
    X_pred = np.where(update_indicator_matrix * X == 0, predicted_value, X)
    return X_pred
    raise NotImplementedError
