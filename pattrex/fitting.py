"""
@motjuste
@kangcifong
"""
import numpy as np
import numpy.polynomial.polynomial as pol
import numpy.linalg as la
from scipy.stats import norm, exponweib, multivariate_normal
from sklearn.linear_model import BayesianRidge


def fit_normal_distribution(data, x_pad=10):
    data_mean = np.mean(data)
    data_std = np.std(data)

    x = np.linspace(data.min() - x_pad, data.max() + x_pad, 100)
    y = norm.pdf(x, data_mean, data_std)
    return data_mean, data_std, x, y


# noinspection PyPep8Naming,PyUnresolvedReferences,PyUnresolvedReferences
def fit_multivariate_normal_dist(data, colwise=True, ddof=None,
                                 get_pdf=True, padding=10, n=100,
                                 X_unknown=None, X_unknown_dim=None):
    if not colwise:
        data = np.transpose(data)
    # find mean and covariance of the data
    data_mean = np.mean(data, axis=1)
    data_cov = np.cov(data, ddof=ddof)

    data_cov_sqrt = np.sqrt(data_cov)
    data_std = np.array([data_cov_sqrt[i, i] for i in range(data.shape[0])])

    # 2D limited code starts here
    assert data.shape[0] == 2, "prediction for unknowns supported only " \
                               "for 2D data"
    data_corr = np.corrcoef(data)[0, 1]
    ret_ = [(data_mean, data_cov, data_corr)]

    # FIXME: @motjuste: asserts for 2D
    data_min = data.min(axis=1)
    data_max = data.max(axis=1)

    x = np.linspace(data_min[0] - padding, data_max[0] + padding, n)
    # noinspection PyUnresolvedReferences
    y_pred = data_mean[1] + (data_corr *
                             (data_std[1]/data_std[0]) *
                             (x - data_mean[0]))

    ret_.append((x, y_pred))

    if X_unknown is not None:
        assert X_unknown_dim is not None, "please provide the dim of the " \
                                     "variable for which the values are " \
                                     "provided"
        assert X_unknown_dim < data.shape[0], "Dimension mismatch; X_unknown " \
                                              "dim > dimension of the data. " \
                                              "Remember, indexing starts at 0"
        assert X_unknown_dim >= 0, "Invalid X_unknown_dim"

        pred_dim = 1 - X_unknown_dim
        Y_pred = data_mean[pred_dim] + data_corr * (data_std[
                                                        pred_dim] / data_std[
                                                        X_unknown_dim]) * (
                                           X_unknown - data_mean[X_unknown_dim])

        ret_.append((X_unknown, Y_pred))

    # generate pdf
    if get_pdf:
        assert data.shape[0] == 2, "generating pdf supported only for 2D data"

        # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
        xy_steps = (data_max - data_min) / (n + 2 * padding)
        # noinspection PyPep8
        x, y = np.mgrid[data_min[0] - padding: data_max[0] + padding:
                            xy_steps[0],
                        data_min[1] - padding: data_max[1] + padding:
                            xy_steps[1]]

        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = multivariate_normal(data_mean, data_cov)

        # use `plt.contourf(x, y, pdf)`
        ret_.append((x, y, rv.pdf(pos)))

    return ret_


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def fit_weibull_distribution(data, init_k=None, init_a=None, n_iter=30,
                             change_thresh=1e-3, return_err=False):
    # noinspection PyPep8Naming,PyShadowingNames
    def delta_1(ka, d, N):
        k = ka[0]
        a = ka[1]

        d_a = d / a

        del_k = (N / k) - (N * np.log(a)) + np.sum(np.log(d)) - \
            np.sum(np.power(d_a, k) * np.log(d_a))
        del_a = (k / a) * (np.sum(np.power(d_a, k)) - N)

        return np.array([1 * del_k, 1 * del_a])

    # noinspection PyPep8Naming,PyShadowingNames,PyShadowingNames
    def delta_2(ka, d, N):
        k = ka[0]
        a = ka[1]

        d_a = d / a

        # noinspection PyPep8
        del2_k = -1 * (N / (k * k)) - \
                 np.sum(np.power(d_a, k) * np.power(np.log(d_a), 2))
        del2_a = (k / a * a) * (N - ((k + 1) * np.sum(np.power(d_a, k))))
        del2_ka = ((1 / a) * np.sum(np.power(d_a, k))) + \
                  ((k / a) * np.sum(np.power(d_a, k) * np.log(d_a))) - N / a

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
    return (k / a) * np.power(data / a, (k - 1)) * np.exp(
            -(np.power(data / a, k)))


def fit_weibull_distribution_sp(data, init_a=1, init_c=1, scale=1, loc=0):
    vals = exponweib.fit(data, init_a, init_c, scale=scale, loc=loc)
    return vals, data, exponweib.pdf(data, *vals)


# # POLYNOMIAL FITTING ##########################

# ## ORDINARY LEAST SQUARES #####################
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
        V_unknown = pol.polyvander(X_unknown, degree)
        y_pred = np.dot(V_unknown, coeff)

        return coeff, (x, y), (X_unknown, y_pred)
    else:
        return coeff, (x, y)


# ## BAYESIAN PARAMETER ESTIMATION ##############

# sklearn approach
# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming
def fit_polynomial_bayesian_skl(X, Y, degree,
                                lambda_shape=1.e-6, lambda_invscale=1.e-6,
                                padding=10, n=100,
                                X_unknown=None):
    X_v = pol.polyvander(X, degree)

    clf = BayesianRidge(lambda_1=lambda_shape, lambda_2=lambda_invscale)
    clf.fit(X_v, Y)

    coeff = np.copy(clf.coef_)

    # there some weird intercept thing
    # since the Vandermonde matrix has 1 at the beginning, just add this
    # intercept to the first coeff
    coeff[0] += clf.intercept_

    ret_ = [coeff]

    # generate the line
    x = np.linspace(X.min()-padding, X.max()+padding, n)
    x_v = pol.polyvander(x, degree)

    # using the provided predict method
    y_1 = clf.predict(x_v)

    # using np.dot() with coeff
    y_2 = np.dot(x_v, coeff)

    ret_.append(((x, y_1), (x, y_2)))

    if X_unknown is not None:
        xu_v = pol.polyvander(X_unknown, degree)

        # using the predict method
        yu_1 = clf.predict(xu_v)

        # using np.dot() with coeff
        yu_2 = np.dot(xu_v, coeff)

        ret_.append(((X_unknown, yu_1), (X_unknown, yu_2)))

    return ret_


# manual approach
# noinspection PyPep8Naming,PyUnresolvedReferences
def fit_polynomial_bayesian(x, y, degree,
                            sig2=None, sig2_0=3.0,
                            use_pinv=False, use_lsmr=False,
                            padding=10, n=100, get_pdf=True,
                            X_unknown=None):
    X = pol.polyvander(x, degree)

    if sig2 is None:
        sig2 = np.var(y)

    Xt = X.T

    prec = (1/sig2) * (np.dot(Xt, X)) + (1/sig2_0) * np.identity(degree + 1)

    # different approaches to inverse calculation
    # TODO: @motjuste: which one is correct
    prec_inv = la.pinv(prec) if use_pinv else la.inv(prec)

    mu = (1/sig2) * np.dot(prec_inv, np.dot(Xt, y))

    # FIXME: @motjuste: choose the one righteous approach
    if use_lsmr:
        from scipy.sparse.linalg import lsmr

        coeff = lsmr(X, y, damp=(sig2/sig2_0))[0]
    else:
        if use_pinv:
            coeff = la.pinv(np.dot(Xt, X) + (sig2/sig2_0) * np.identity(
                    degree + 1))
        else:
            coeff = la.inv(np.dot(Xt, X) + (sig2/sig2_0) * np.identity(
                    degree + 1))

        coeff = np.dot(coeff, np.dot(Xt, y))

    ret_ = [coeff]

    # generate the line
    x = np.linspace(x.min()-padding, x.max()+padding, n)
    x_v = pol.polyvander(x, degree)

    # the mean of the posterior of y is the best prediction
    y_1 = np.dot(x_v, mu)

    # using np.dot() with coeff
    y_2 = np.dot(x_v, coeff)

    ret_.append(((x, y_1), (x, y_2)))

    if X_unknown is not None:
        xu_v = pol.polyvander(X_unknown, degree)

        # the mean of the posterior of y is the best prediction
        yu_1 = np.dot(xu_v, mu)

        # using np.dot() with coeff
        yu_2 = np.dot(xu_v, coeff)

        ret_.append(((X_unknown, yu_1), (X_unknown, yu_2)))

    if get_pdf:
        # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
        data_min = np.array([x.min(), y.min()])
        data_max = np.array([x.max(), y.max()])
        xy_steps = (data_max - data_min) / (n + 2 * padding)
        # noinspection PyPep8
        xx, yy = np.mgrid[data_min[0] - padding: data_max[0] + padding:
        xy_steps[0],
               data_min[1] - padding: data_max[1] + padding:
               xy_steps[1]]

        x = xx[:, 0]
        y = yy[0, :]
        pdf = []

        for i, x_ in enumerate(x):
            x_v = pol.polyvander(x_, degree).T

            mean = np.dot(mu.T, x_v)
            var = sig2 + np.dot(x_v.T, np.dot(prec_inv, x_v))
            pdf_ = norm.pdf(y, mean, var).T
            pdf.append(pdf_)

        pdf = np.array(pdf)[:, :, 0]

        # use `plt.contourf(x, y, pdf)`
        ret_.append((xx, yy, pdf))

    return ret_




