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


if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python gaussianize.py test_data.csv
    import csv
    import sys, os
    import traceback
    from optparse import OptionParser, OptionGroup

    parser = OptionParser(usage="usage: %prog [options] data_file.csv \n"
                                "It is assumed that the first row and first column of the data CSV file are labels.\n"
                                "Use options to indicate otherwise.")
    group = OptionGroup(parser, "Input Data Format Options")
    group.add_option("-c", "--no_column_names",
                     action="store_true", dest="nc", default=False,
                     help="We assume the top row is variable names for each column. "
                          "This flag says that data starts on the first row and gives a "
                          "default numbering scheme to the variables (1,2,3...).")
    group.add_option("-r", "--no_row_names",
                     action="store_true", dest="nr", default=False,
                     help="We assume the first column is a label or index for each sample. "
                          "This flag says that data starts on the first column.")
    group.add_option("-d", "--delimiter",
                     action="store", dest="delimiter", type="string", default=",",
                     help="Separator between entries in the data, default is ','.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="gaussian_output.csv",
                     help="Where to store gaussianized data.")
    group.add_option("-q", "--qqplots",
                     action="store_true", dest="q", default=False,
                     help="Produce qq plots for each variable before and after transform.")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if not len(args) == 1:
        print "Run with '-h' option for usage help."
        sys.exit()

    #Load data from csv file
    filename = args[0]
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=options.delimiter)
        if options.nc:
            variable_names = None
        else:
            variable_names = reader.next()[(1 - options.nr):]
        sample_names = []
        data = []
        for row in reader:
            if options.nr:
                sample_names = None
            else:
                sample_names.append(row[0])
            data.append(row[(1 - options.nr):])

    try:
        X = -np.array(data, dtype=float)  # Data matrix in numpy format
    except:
        print "Incorrect data format.\nCheck that you've correctly specified options " \
              "such as continuous or not, \nand if there is a header row or column.\n" \
              "Run 'python gaussianize.py -h' option for help with options."
        traceback.print_exc(file=sys.stdout)
        sys.exit()

    out = Lambert()
    y = out.fit_transform(X)
    with open(options.output, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=options.delimiter)
        if not options.nc:
            writer.writerow([""] * (1 - options.nr) + variable_names)
        for i, row in enumerate(y):
            if not options.nr:
                writer.writerow([sample_names[i]] + list(row))
            else:
                writer.writerow(row)

    if options.q:
        print 'Making qq plots'
        prefix = options.output.split('.')[0]
        if not os.path.exists(prefix+'_q'):
            os.makedirs(prefix+'_q')
        out.qqplot(X, prefix=prefix + '_q/q')