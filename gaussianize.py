"""
Transform samples of heavy-tailed data so that it is approximately normally distributed

Based on the paper:
The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h transformation as a special case
Based on algorithm in Appendix C
Author generously provides code in R:
https://cran.r-project.org/web/packages/LambertW/

This code written by Greg Ver Steeg, 2015.
"""

import numpy as np
from scipy.special import lambertw
from scipy.stats import kurtosis
from scipy.optimize import fmin  # TODO: Explore efficacy of other opt. methods

np.seterr(all='warn')

class Lambert(object):
    """
    Gaussianize heavy-tailed data using Lambert's W.

    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.

    Parameters
    ----------
    tol : float, default = 1e-4

    max_iter : int, default = 200
        Maximum number of iterations to search for correct parameters of Lambert transform.

    Attributes
    ----------
    taus : list of tuples
        For each variable, we have a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.

    References
    ----------
    [1] Georg Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
    """

    def __init__(self, tol=1.22e-4, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.taus = []  # Store tau for each transformed variable

    def fit(self, x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."
        for x_i in x.T:
            self.taus.append(igmm(x_i, tol=self.tol, max_iter=self.max_iter))

    def transform(self, x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."
        if x.shape[1] != len(self.taus):
            print "%d variables in test data, but %d variables were in training data." % (x.shape[1], len(self.taus))

        return np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.taus)]).T

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def invert(self, y):
        return np.array([inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.taus)]).T

    def qqplot(self, x, prefix='qq'):
        """Show qq plots compared to normal before and after the transform."""
        import pylab
        from scipy.stats import probplot
        y = self.transform(x)

        for i, (x_i, y_i) in enumerate(zip(x.T, y.T)):
            probplot(x_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + '_%d_before.png' % i)
            pylab.clf()

            probplot(y_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + '_%d_after.png' % i)
            pylab.clf()


def w_d(z, delta):
    # Eq. 9
    if delta < 1e-6:
        return z
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)


def w_t(y, tau):
    # Eq. 8
    return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])


def inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))


def igmm(y, tol=1.22e-4, max_iter=100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    delta0 = delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = delta_gmm(z)
        x = tau0[0] + tau1[1] * w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break
        else:
            if k == max_iter - 1:
                print "Warning: No convergence after %d iterations. Increase max_iter." % max_iter
    return tau1


def delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z)

    def func(q):
        u = w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        else:
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)


def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all='ignore'):
        delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.25)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0