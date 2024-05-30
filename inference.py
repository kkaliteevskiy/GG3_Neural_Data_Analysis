# Credit: Functions here are essentially copies of those in the
# SSM package by Scott Linderman et al. https://github.com/lindermanlab/ssm

import numba
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp as logsumexp_scipy
from scipy.special import gammaln

LOG_EPS = 1e-16


@numba.jit(nopython=True, cache=True)
def logsumexp(x):
    N = x.shape[0]

    # find the max
    m = -np.inf
    for i in range(N):
        m = max(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += np.exp(x[i] - m)

    return m + np.log(out)


@numba.jit(nopython=True, cache=True)
def dlse(a, out):
    K = a.shape[0]
    lse = logsumexp(a)
    for k in range(K):
        out[k] = np.exp(a[k] - lse)


@numba.jit(nopython=True, cache=True)
def forward_pass(pi0,
                 Ps,
                 log_likes,
                 alphas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    alphas[0] = np.log(pi0) + log_likes[0]
    for t in range(T-1):
        m = np.max(alphas[t])
        alphas[t+1] = np.log(np.dot(np.exp(alphas[t] - m), Ps[t * hetero])) + m + log_likes[t+1]
    return logsumexp(alphas[T-1])


@numba.jit(nopython=True, cache=True)
def backward_pass(Ps,
                  log_likes,
                  betas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    tmp = np.zeros(K)

    # Initialize the last output
    betas[T-1] = 0
    for t in range(T-2,-1,-1):
        tmp = log_likes[t+1] + betas[t+1]
        m = np.max(tmp)
        betas[t] = np.log(np.dot(Ps[t * hetero], np.exp(tmp - m))) + m


def hmm_normalizer(pi0, Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    if Ps.ndim == 2:
        Ps = Ps[None, :, :]
    assert Ps.ndim == 3

#     # Make sure everything is C contiguous
#     pi0 = to_c(pi0)
#     Ps = to_c(Ps)
#     ll = to_c(ll)

    forward_pass(pi0, Ps, ll, alphas)
    return logsumexp(alphas[-1])


def hmm_expected_states(pi0, Ps, ll, filter=False):
    """
    Calculates the posterior probabilities of HMM states given the observations, implicitly input via
    the matrix of observation log-likelihoods.
    :param pi0: shape (K,), vector of initial state probabilities.
    :param Ps: shape (K, K): state transition matrix (time-homogeneous case), or:
               shape (T-1, K, K): temporal sequence

    :param ll: shape (T, K): matrix of log-likelihoods (i.e. log observation probabilities
                             evaluated for the actual observations).
    :param filter: False by default. If True the function calculates the so-called "filtered"
                   posterior probabilities which only take into account observations until time t
                   (as opposed to all observations until time T (with Python index T-1), which is what
                   is calculated by default).
    :return:
    expected_states: this is an array of shape (T, K) with the t-th row giving the
                     posterior probabilities of the different Markov states,
                     conditioned on the sequence of observations.
    normalizer: this is the model log-likelihood, i.e. it is the log-probability of the entire sequence
                of observations (given the model parameters, which are implicit here).
    """
    T, K = ll.shape

    if Ps.ndim == 2:
        Ps = Ps[None, :, :]
    assert Ps.ndim == 3

    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)
    normalizer = logsumexp(alphas[-1])

    betas = np.zeros((T, K))
    if not filter:
        backward_pass(Ps, ll, betas)

    # Compute P[x_t | n_{1:T}] for t = 1, ..., T (if filter = True, calculate P[x_t | n_{1:t}] instead).
    expected_states = alphas + betas
    expected_states -= logsumexp_scipy(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)

    # expected_joints calculation removed

    return expected_states, normalizer


@numba.jit(nopython=True, cache=True)
def backward_sample(Ps, log_likes, alphas, us, zs):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K
    assert us.shape[0] == T
    assert zs.shape[0] == T

    lpzp1 = np.zeros(K)
    lpz = np.zeros(K)

    # Trick for handling time-varying transition matrices
    hetero = (Ps.shape[0] == T-1)

    for t in range(T-1,-1,-1):
        # compute normalized log p(z[t] = k | z[t+1])
        lpz = lpzp1 + alphas[t]
        Z = logsumexp(lpz)

        # sample
        acc = 0
        zs[t] = K-1
        for k in range(K):
            acc += np.exp(lpz[k] - Z)
            if us[t] < acc:
                zs[t] = k
                break

        # set the transition potential
        if t > 0:
            lpzp1 = np.log(Ps[(t-1) * hetero, :, int(zs[t])] + LOG_EPS)


@numba.jit(nopython=True, cache=True)
def _hmm_sample(pi0, Ps, ll):
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Sample backward
    us = npr.rand(T)
    zs = -1 * np.ones(T)
    backward_sample(Ps, ll, alphas, us, zs)
    return zs


def hmm_sample(pi0, Ps, ll):
    return _hmm_sample(pi0, Ps, ll).astype(int)


@numba.jit(nopython=True, cache=True)
def _viterbi(pi0, Ps, ll):
    """
    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)
    if not hetero:
        assert Ps.shape[0] == 1

    # Pass max-sum messages backward
    scores = np.zeros((T, K))
    args = np.zeros((T, K))
    for t in range(T-2,-1,-1):
        vals = np.log(Ps[t * hetero] + LOG_EPS) + scores[t+1] + ll[t+1]
        for k in range(K):
            args[t+1, k] = np.argmax(vals[k])
            scores[t, k] = np.max(vals[k])

    # Now maximize forwards
    z = np.zeros(T)
    z[0] = (scores[0] + np.log(pi0 + LOG_EPS) + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, int(z[t-1])]

    return z


def viterbi(pi0, Ps, ll):
    """
    Find the most likely state sequence
    """
    return _viterbi(pi0, Ps, ll).astype(int)


def poisson_logpdf(counts, lambdas, mask=None):
    """
    Compute the log probability of a Poisson distribution.
    This will broadcast as long as data and lambdas have the same
    (or at least compatible) leading dimensions.
    Parameters
    ----------
    counts : array_like of shape (T,) or (Ntrials, T),
             array of integer counts for which to evaluate the log probability
    lambdas : array_like of shape (K,)
        The rates (mean counts) of the Poisson distribution(s)
    Returns
    -------
    lls : array_like with shape (T, K), or (Ntrials, T, K) depending on
          the shape of 'counts'.
        Log probabilities under the Poisson distribution(s).
    """
    assert counts.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert counts.ndim == 1 or counts.ndim == 2
    if counts.ndim == 1:
        counts = counts[:, None]
    elif counts.ndim == 2:
        counts = counts[:,:,None]

    # Compute log pdf
    lambdas[lambdas == 0] = 1e-8
    lls = -gammaln(counts + 1) - lambdas + counts * np.log(lambdas)
    return lls