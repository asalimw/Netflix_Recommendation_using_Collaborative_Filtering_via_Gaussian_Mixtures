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

    n, d = X.shape  #
    mu, var, p = mixture  # Unpacking the mixture
    K, _ = mixture.mu.shape

    # Compute weights
    f = np.zeros((n, K))  # Initialise Log Likelihood as 0

    for i in range(n):
        for j in range(K):
            # Multivariate Gaussian, N(x; mu, var*I)
            # With Bayes Rule - Check wikipedia for the formula
            # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

            sigma = var[j] * np.identity(d)
            g_numerator = (-1 / 2) * ((X[i] - mu[j]).T.dot(np.linalg.inv(sigma))).dot((X[i] - mu[j]))
            g_denominator = (((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2)))
            gaussian = np.exp(g_numerator) / g_denominator

            # Calculate the f(u,j)/f(i,j) - the MLE for Gaussian Distribution
            # f_ui[i, j] = np.log(pi[j] + 1e-16) + log_pdf[i, j]
            f[i, j] = p[j] * gaussian


    f += np.log(p + 1e-16)  # 1e-e16 is to avoid numerical underflow


    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1, 1)  # Store this to calculate log_lh
    log_post = f - logsums

    log_likelihood = np.sum(logsums, axis=0).item()    # # This is the log likelihood


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

    return log_post, log_likelihood

    # Create a sparse matrix to indicate where X is non-zero, which will help us pick Cu indices
    # sparse_matrix = np.where(X > 0, 1, 0)



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
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
