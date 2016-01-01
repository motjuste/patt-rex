"""
@motjuste
@kangcifong
"""
import numpy as np
import numpy.polynomial.polynomial as pol
import numpy.linalg as la
from scipy.stats import norm, exponweib


def fit_normal_distribution(data, x_pad=10):
    data_mean = np.mean(data)
    data_std = np.std(data)

    x = np.linspace(data.min()-x_pad, data.max()+x_pad, 100)
    y = norm.pdf(x, data_mean, data_std)
    return data_mean, data_std, x, y


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def fit_weibull_distribution(data, init_k=None, init_a=None, n_iter=30,
                             change_thresh=1e-3, return_err=False):

    # noinspection PyPep8Naming,PyShadowingNames
    def delta_1(ka, d, N):
        k = ka[0]
        a = ka[1]

        d_a = d/a

        del_k = (N/k) - (N*np.log(a)) + np.sum(np.log(d)) - \
            np.sum(np.power(d_a, k) * np.log(d_a))
        del_a = (k/a) * (np.sum(np.power(d_a, k)) - N)

        return np.array([1*del_k, 1*del_a])

    # noinspection PyPep8Naming,PyShadowingNames,PyShadowingNames
    def delta_2(ka, d, N):
        k = ka[0]
        a = ka[1]

        d_a = d/a

        del2_k = -1 * (N/(k*k)) - \
            np.sum(np.power(d_a, k) * np.power(np.log(d_a), 2))
        del2_a = (k/a*a) * (N - ((k + 1) * np.sum(np.power(d_a, k))))
        del2_ka = ((1/a) * np.sum(np.power(d_a, k))) + \
            ((k/a) * np.sum(np.power(d_a, k) * np.log(d_a))) - N/a

        return np.array([[del2_k, del2_ka], [del2_ka, del2_a]])

    N = data.size
    k_ = 1
    a_ = np.mean(data)
    if init_k is not None:
        k_ = init_k

    if init_a is not None:
        a_ = init_a

    ka = np.array([k_, a_]).astype(np.float)

    ka_err = np.array([np.inf, np.inf])
    thresh = np.ones(2) * change_thresh
    max_iter = n_iter
    i = 0

    err = []

    while (np.any(ka_err > thresh)) and i < max_iter:
        i += 1

        del_1 = delta_1(ka, data, N)
        del_2 = delta_2(ka, data, N)

        ka_err = np.dot((np.linalg.inv(del_2)), del_1)

        ka -= ka_err

        err.append(ka_err)
        ka_err = abs(ka_err)

    if return_err:
        res = (ka[0], ka[1], data, weib_pdf(data, ka[0], ka[1]), err)
    else:
        res = (ka[0], ka[1], data, weib_pdf(data, ka[0], ka[1]))

    return res


def weib_pdf(data, k, a):
    return (k/a) * np.power(data/a, (k-1)) * np.exp(-(np.power(data/a, k)))


def fit_weibull_distribution_sp(data, init_a=1, init_c=1, scale=1, loc=0):
    vals = exponweib.fit(data, init_a, init_c, scale=scale, loc=loc)
    return vals, data, exponweib.pdf(data, *vals)


# # POLYNOMIAL FITTING ##########################
# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def fit_polynomial_nplstsq(X, Y, degree, x_pad=10, X_unknown=None):
    """
    Fits a polynomial of degree with variables X over Y

    :param X: variables. n dimensional array, with row-wise instance
    :param Y: value to fit to. n dimensional array, with row wise instances
    :param degree: degree of polynomial. list, dim(X) = len(degree)
    :param x_pad: padding around X for generated data for the model
    :param X_unknown: unknown X for which y is to be predicted, if not None.
    :return: W, coefficients.
    :return: (x, y): (x, y) values of the generated data for the model.
    """
    # TODO: tests for truly n-dimensional X
    # construct the appropriate Vandermonde matrix
    V = pol.polyvander(X, degree)

    # get coefficients by least squares
    coeff = la.lstsq(V, Y)[0]  # discarding other information

    # construct some fitting data
    x = np.linspace(X.min() - x_pad, X.max() + x_pad, 100)
    x_ = pol.polyvander(x, degree)

    y = np.dot(x_, coeff)

    # calculate y for unknown X, if provided
    if X_unknown is not None:
        V_unknown = pol.polyvander(X_unknown)
        y_pred = np.dot(V_unknown, coeff)

        return coeff, y_pred, (x, y)
    else:
        return coeff, (x, y)
