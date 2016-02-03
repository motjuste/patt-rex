"""
@motjuste
Dimensionality Reduction
"""
import numpy as np
import numpy.linalg as la


# # PRINCIPAL COMPONENT ANALYSIS ##############################################
def pca(X_, k=None, err=0.0, use_eigh=True):
    assert err >= 0, "err is less than 0"
    assert err <= 1, "err is greater than 1"

    if k is not None:
        assert k <= X_.shape[0], "k is greater than dim of X_"

    # Normalize data
    X_mean = X_.mean(axis=1).reshape(X_.shape[0], 1)
    X = X_ - np.tile(X_mean, (1, X_.shape[1]))

    # Calculate the covarianve matrix
    C = np.cov(X)

    # Do Eigen Analysis
    if use_eigh:
        evals, evects = la.eigh(C)

        # The evals are sorted ascending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evects = evects[:, idx]
    else:
        evals, evects = la.eig(C)

    sum_evals = np.cumsum(evals)

    if k is None:
        err_ = 1 - err

        k = np.searchsorted(sum_evals, sum_evals[-1] * err_, side='left') + 1

        print("Found for error {}%,"
              "k can be at least {}".format(err * 100, k))

    else:
        err_ = sum_evals[k - 1] / sum_evals[-1]

        print("Found that choosing k as {}"
              " will lead to at most error {}%".format(k, (1 - err_) * 100))

    # Choose the evects for the k
    evects_k = evects[:, :k]

    # project the normalized data in k dim
    projection = (X.T).dot(evects_k)

    return projection, (C), (evals, evects)


# # FISHERS'S LDA #############################################################
def lda(X, y, k=None, err=0.0, use_eigh=True, ddof=0):
    # find the number of classes and corresponding datas and their mean and cov
    classes = np.unique(y)
    X_mean = np.mean(X, axis=1)
    dim = X.shape[0]

    S_w = np.zeros((dim, dim))
    S_b = np.zeros((dim, dim))

    datas = dict()
    for c in classes:
        datas[c] = dict()
        X_c = X[:, y == c]
        n = X_c.shape[1]

        m = np.mean(X_c, axis=1)
        m = m.reshape(m.shape[0], 1)

        c_ = np.cov(X_c - np.tile(m, (1, n)), ddof=ddof)

        m_diff = (m - X_mean).reshape(dim, 1)

        datas[c]["X"] = X_c
        datas[c]["mean"] = m
        datas[c]["n"] = n
        datas[c]["cov"] = c_

        S_w += c_
        S_b += m_diff.dot(m_diff.T)

    print("Found {} classes of {} dimensional data".format(
        len(classes), dim))
    print("\n".join("Class {}: {} samples".format(k, v["n"])
                    for k, v in datas.items()))

    W = la.inv(S_w).dot(S_b)

    # do eigen analysis
    if use_eigh:
        evals, evects = la.eigh(W)

        # The evals are sorted ascending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evects = evects[:, idx]
    else:
        evals, evects = la.eig(W)

    sum_evals = np.cumsum(evals)

    if k is None:
        err_ = 1 - err

        k = np.searchsorted(sum_evals, sum_evals[-1] * err_, side='left') + 1

        print("Found for error {}%,"
              "k can be at least {}".format(err * 100, k))

    else:
        err_ = sum_evals[k - 1] / sum_evals[-1]

        print("Found that choosing k as {}"
              " will lead to at most error {}%".format(k, (1 - err_) * 100))

    # Choose the evects for the k
    evects_k = evects[:, :k]

    # project the normalized data in k dim
    projection = (X.T).dot(evects_k)

    return projection, (S_w, S_b, W), (evals, evects)
