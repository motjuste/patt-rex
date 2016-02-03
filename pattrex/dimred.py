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
    X_mean = X_.mean()
    X = X_ - X_mean

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

    return projection, C, (evals, evects)


# # FISHERS'S LDA #############################################################
def lda(X, y, k=None, err=0.0, use_eigh=True):
    pass
