import numpy as np
import scipy.stats as stats
from scipy.stats import anderson, shapiro
import pylab
import sys
sys.path.append('..')
import gaussianize as g

ns = 1000  # number of samples to use in tests
experimental_data = [-0.594,0.666,-1.240,-0.789,-1.131,1.811,0.275,0.522,-0.494,0.236,0.808,1.143,0.655,-0.904,-1.161,1.137,-0.965,-0.970,1.762,-1.701,1.056,0.341,-0.258,0.260,0.256,-0.465,0.342,-0.588,-0.593,-1.418,0.264,-1.628,-1.386,0.051,-1.382,-0.979,0.867,0.624,0.868,-1.381,-1.290,-0.055,-1.122,0.614,1.104,1.446,-0.454,0.114,1.003,0.849,-0.856,0.670,0.011,0.673,-0.385,1.428,-1.419,-0.887,1.122,-0.653,0.000,1.557,0.462,0.638,-0.344,1.585,0.849,1.545,2.178,-1.062,-0.327,-1.628,-0.571,-0.771,0.389,-0.297,-0.004,-1.098,-1.364,-0.217,-0.197,1.574,-0.009,0.874,-1.436,1.490,-1.704,-0.192,0.639,0.248,-1.489,-1.252,1.062,1.167,0.795,-0.842,0.031,0.776,1.084,0.081,-0.438,-0.550,-1.335,-0.759,-0.108,1.034,1.419,1.501,0.148,1.528,1.025,0.369,1.002,-0.636,0.946,1.168,1.299,0.052,-0.304,-1.325,-0.902,0.720,-1.436,0.448,-1.181,-0.897,-1.438,0.303,1.794,1.291,-0.992,0.384,-0.366,-0.831,-0.232,1.091,-0.297,1.204,-1.212,-0.624,1.463,1.388,0.530,-0.405,-0.536,0.658,-0.952,0.847,0.229,-0.756,2.046,-1.333,-1.145,0.053,1.016,-0.814,-1.239,1.801,-0.189,0.189,0.884,-1.095,0.269,0.666,-0.044,1.595,1.656,1.047,-0.863,-1.655,-1.317,-0.505,1.501,-0.793,-0.236,0.763,-1.043,1.385,-0.380,-1.086,-0.452,-1.054,1.562,0.581,0.333,0.611,1.556,-0.565,1.119,0.973,-0.590,-1.340,-1.267,1.687,1.239,1.245,-0.036,1.023,-0.885,-0.962,-0.183,-1.096,1.391,-0.523,1.204,0.868,1.169,1.459,0.100,-1.019,-0.144,-1.070,1.312,-1.505,-1.570,-0.099,0.978,-0.870,-0.767,0.306,-1.308,-1.049,-0.442,1.565,0.680,0.948,-0.716,0.072,0.832,-0.859,-0.533,0.098,1.654,1.687,1.082,0.068,-0.420,1.027,0.703,0.473,1.395,-1.698,1.542,-1.307,-0.985,-1.161,-0.835,1.364,0.706,-0.480,-0.904,-1.113,0.835,-0.821,1.370,-0.454,0.753,-1.258,0.109,-0.688,0.221,-1.110,-0.383,0.507,-0.612,-1.224,1.472,0.484,-1.055,-0.826,-0.786,1.125,-1.068,0.841,0.892,-0.499,-1.341,-0.843,-0.837,0.107,1.358,-0.660,-1.100,1.382,-0.612,1.698,0.409,-0.618,0.292,-0.916,-0.808,-0.064,-0.776,-1.545,1.575,-1.501,-1.469,-1.440,-0.001,-1.146,0.979,1.176,-1.011,1.592,-0.567,0.224,-0.017,1.349,-1.015,0.240,1.401,1.537,0.234,0.808,0.436,0.554,1.662,-1.615,-1.513,0.174,-0.849,-0.297,-1.476,1.212,-0.978,2.108,0.221,1.371,-0.427,-1.595,0.359,-1.219,-1.669,-0.200,0.173,0.248,-0.835,-1.209,1.325,1.000,-0.077,-0.630,0.424,-1.570,-0.164,0.128,-0.945,0.141,1.854,0.889,0.915,-0.603,1.071,-0.120,1.323,0.888,-0.349,0.035,-0.622,-1.229,1.415,0.631,1.188,-0.460,0.518,-0.313,1.702,-0.993,-0.529,-0.626,-1.014,1.725,-0.098,1.132,-0.117,-1.447,0.162,1.426,-0.996,-0.267,-1.025,-0.283,-0.953,-0.589,1.237,-0.607,-0.398,-0.583,-0.363,-0.453,-1.100,-0.743,1.093,-1.549,-0.704,-0.479,-0.362,1.115,-0.577,-0.549,0.310,-1.084,-0.833,-0.827,1.850,-1.398,-0.502,-1.583,-1.517,1.209,-1.637,0.419,1.583,-0.576,0.731,1.208,0.308,0.947,-1.303]

def test_invert():
    # Test if we can transform and transform back
    mu, sigma, delta = 0.5, 1.7, 0.33
    x = np.random.normal(loc=mu, scale=sigma, size=ns)

    y = g.inverse(x, (mu, sigma, delta))
    x_prime = g.w_t(y, (mu, sigma, delta))
    assert np.allclose(x, x_prime)


def test_recover():
    # Generate data from a normal, then apply the W transform, and check if we can recover parameters
    # See Table 2 of Georg's paper
    # These results seem a little worse, but we'd have to run many replications to directly compare.
    mu, sigma = 0, 1
    print ('del_true\tns\tmu\tsigma\tdelta').expandtabs(10)
    for delta in [0, 1./3, 1., 1.5]:
        for n in [50, 100, 1000]:
            x = np.random.normal(loc=mu, scale=sigma, size=n)
            y = g.inverse(x, (mu, sigma, delta))
            mu_prime, sigma_prime, delta_prime = g.igmm(y)
            print ('%0.3f\t%d\t%0.3f\t%0.3f\t%0.3f' % (delta, n, mu_prime, sigma_prime, delta_prime)).expandtabs(10)
            assert np.abs(mu - mu_prime) < 10. / np.sqrt(n)
            assert np.abs(sigma - sigma_prime) < 10. / np.sqrt(n)
            assert np.abs(delta - delta_prime) < 10. / np.sqrt(n)


def test_normality_increase():
    # Generate random data and check that it is more normal after inference
    for i, y in enumerate([np.random.standard_cauchy(size=ns), experimental_data]):
        print 'Distribution %d' % i
        print 'Before'
        print ('anderson: %0.3f\tshapiro: %0.3f' % (anderson(y)[0], shapiro(y)[0])).expandtabs(30)
        stats.probplot(y, dist="norm", plot=pylab)
        pylab.savefig('%d_before.png' % i)
        pylab.clf()

        tau = g.igmm(y)
        x = g.w_t(y, tau)
        print 'After'
        print ('anderson: %0.3f\tshapiro: %0.3f' % (anderson(x)[0], shapiro(x)[0])).expandtabs(30)
        stats.probplot(x, dist="norm", plot=pylab)
        pylab.savefig('%d_after.png' % i)
        pylab.clf()


def test_transform():
    x = np.hstack([np.random.standard_cauchy(size=(1000, 2)), np.random.normal(size=(1000, 2))])
    out = g.Lambert()
    out.fit(x)
    y = out.transform(x)
    x_prime = out.invert(y)
    assert np.allclose(x_prime, x)
    out.qqplot(x)
