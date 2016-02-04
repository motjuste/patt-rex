"""
@aksakalli
@Cifong Kang
Here is a draft implementation of k-means algorithms.
You need "img" folder under the same path in order to export plot images.

!!! Important: If you want to compare runtime performance,
comment out all code block which are responsible to plot

If you want to merge png files as gif animation use "convert" tool:

$ convert -delay 10 -loop 0 macqueen-*.png macqueen.gif

or you can achieve this with pyplot.

See exported animations:
MacQueen's algorithm - http://imgur.com/ZT9ftCk
Hartigans's algorithm - http://imgur.com/RhRQ53u
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import distance as spdist

#DATA_PATH = "../data/data-clustering-1.csv"

#data = np.genfromtxt(DATA_PATH, delimiter=",")
#data = np.transpose(data)

#k = 3
np.random.seed(9000)


def compute_centroids(data, idx, k):
    centroids = np.zeros((k, 2))
    for i in range(k):
        centroids[i] = data[idx == i, :].mean(axis=0)
    return centroids


def compute_objective(data, idx, k, centroids):
    total_error = 0
    for i in range(k):
        # based on objective function, compute the error
        total_error += np.linalg.norm(data[idx == i, :] - centroids[i], axis=1).sum()
    return total_error


def show_plotted_cluster(data, idx, centroids, title, k):
    plt.cla()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=idx, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(k))
    plt.title(title)
    plt.show()
    return


def save_plotted_cluster(data, idx, centroids, title, file_prefix, iteration, k):
    plt.cla()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=idx, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(k))
    plt.title(title)
    plt.savefig('img/' + file_prefix + '-' + str(iteration).zfill(5) + '.png')
    return


# importan! comment out save_ploted_cluster calls for performance test

def kmeans_Lloyd(data, k, centroids, save_plot=True):

    ## Important!! the input parameter is different to others, this requires a initial centroid

    # Set t=0 and initialize centroids
    iteration = 0
    maxIteration = 200
    prefix = 'Lloyd'
    title = "Lloyd's algorithm, update: {}"

    idx = np.zeros(len(data))
    # initial plot
    if save_plot:
        save_plotted_cluster(data, idx, centroids,
                             title.format(iteration),
                             prefix, iteration, k)

    # repeat until convergence
    converged = False
    while not converged:

        idxPrev = np.copy(idx)
        # update all clusters
        for i in range(len(data)):
            candidate_class = 0
            candidate_distance = np.inf
            for j in range(k):
                distance = np.linalg.norm(centroids[j] - data[i, :])
                if distance < candidate_distance:
                    candidate_class = j
                    candidate_distance = distance
            idx[i] = candidate_class

        # update all cluster centroid
        centroidsPrev = centroids
        centroids = compute_centroids(data, idx, k)

        # increase iteration counter
        iteration += 1
        # save the current progress
        if save_plot:
            save_plotted_cluster(data, idx, centroids,
                                 title.format(iteration),
                                 prefix, iteration, k)

        if(iteration >= maxIteration):
            converged = True
            convCondition = 1
        if(np.array_equal(idxPrev,idx)):
            converged = True
            convCondition = 2
        if(np.allclose(centroids,centroidsPrev)):
            converged = True
            convCondition = 3

    return centroids, idx, convCondition

def kmeans_hartigans(data, k, save_plot=True, show_plot=True):
    # plotting purpose
    iteration = 0
    prefix = 'hartigan'
    title = "Hartigan's algorithm, update: {}"

    # randomly assign all data to a cluster
    idx = np.random.randint(k, size=len(data))
    # compute the initial centroids
    centroids = compute_centroids(data, idx, k)

    # initial plot
    if save_plot:
        save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration, k)
    while True:
        converged = True
        for i in range(len(data)):

            initial_class = idx[i]
            candidate_class = 0
            candidate_error = np.inf
            for j in range(k):
                idx[i] = j
                centroids = compute_centroids(data, idx, k)
                objective_error = compute_objective(data, idx, k, centroids)
                if (objective_error < candidate_error):
                    candidate_class = j
                    candidate_error = objective_error

            if initial_class != candidate_class:
                converged = False

                # plotting
                centroids = compute_centroids(data, idx, k)
                idx[i] = candidate_class
                iteration += 1
                if save_plot:
                    save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration, k)

            idx[i] = candidate_class

        if converged:
            break

    return centroids, idx


def kmeans_macqueen(data, k, save_plot=True):
    # plotting purpose
    iteration = 0
    prefix = 'macqueen'
    title = "MacQueen's algorithm, update: {}"
    colors = np.zeros((len(data), 3))

    centroids = np.zeros((k, 2))
    n = np.zeros((k, 1))
    idx = np.zeros(len(data))

    for i in range(len(data)):
        candidate_class = 0
        candidate_distance = np.inf
        for j in range(k):
            distance = np.linalg.norm(centroids[j] - data[i, :])
            if distance < candidate_distance:
                candidate_class = j
                candidate_distance = distance

        n[candidate_class] += 1
        centroids[candidate_class] += 1 / n[candidate_class] * (data[i, :] - centroids[candidate_class])

        iteration += 1
        if save_plot:
            save_plotted_cluster(data[0:i + 1], colors[0:i + 1], centroids, title.format(iteration), prefix, iteration,k)

    for i in range(len(data)):
        candidate_class = 0
        candidate_distance = np.inf
        for j in range(k):
            distance = np.linalg.norm(centroids[j] - data[i, :])
            if distance < candidate_distance:
                candidate_class = j
                candidate_distance = distance
        idx[i] = candidate_class

    iteration += 1
    if save_plot:
        save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration,k)

    return centroids, idx


def sqnorm(x):
    return np.power(np.linalg.norm(x), 2)


def hartigan2(X, k, seed=9000):
    nX, mX = X.shape
    np.random.seed(seed)
    y = np.random.randint(k, size=nX)

    m = np.zeros((k, mX))
    e = np.ones(k)
    n = np.zeros(k)
    for kk in range(k):
        Xkk = X[y == kk]
        m[kk] = np.mean(Xkk, axis=0)
        e[kk] = np.sum(sqnorm(Xkk - m[kk]))
        n[kk] = len(Xkk)

    converged = False
    t, t_max = 0, 1000
    while not converged and t < t_max:
        t += 1
        converged = True
        for i, x in enumerate(X):
            ki = y[i]
            y[i] = -1
            Xki = np.copy(X[y == ki])
            n[ki] = len(Xki)

            m[ki] = (n[ki] / (n[ki] - 1)) * (m[ki] - x / n[ki])

            normxki = sqnorm(x - m[ki])
            eki = e[ki]
            e[ki] = np.sum(sqnorm(Xki - m[ki])) - normxki
            ediffki = e[ki] - eki

            ediff = []
            for kk in range(k):
                if kk == ki:
                    ediff.append(ediffki + normxki)
                else:
                    ediff.append(ediffki + sqnorm(x - m[kk]))

            kw = np.argmin(ediff)

            converged = ki == kw

            y[i] = kw
            Xkw = np.copy(X[y == kw])
            n[kw] = len(Xkw)
            m[kw] += (1/n[kw]) * (x - m[kw])
            e[kw] = np.sum(sqnorm(Xkw - m[kw]))

    if not converged:
        raise UserWarning("Did not Converge after {} iterations".format(t))

    return m, y


def lloyd2(data, init_cent, metric='e', verbose=False):
    k = init_cent.shape[0]
    cent = np.copy(init_cent)
    labels = spdist.cdist(data, cent, metric).argmin(axis=1)
    converged = False
    t, tmax = 0, 1000

    while not converged and t < tmax:
        t += 1
        converged = True

        cent_ = np.array([np.mean(data[labels == l], axis=0)
                         for l in range(k)])

        labels_ = spdist.cdist(data, cent_, metric).argmin(axis=1)

        if not np.allclose(cent_, cent) or \
                not np.alltrue(labels == labels_):
            converged = False
            labels = labels_
            cent = cent_

    if not converged:
        # raise UserWarning("did not converge after {} iterations".format(t))
        print("did not converge after {} iterations".format(t))
    elif verbose:
        print("Converged after {} iterations".format(t))

    return cent, labels


def mcqueen2(data, k, metric='e'):
    nX, mX = data.shape

    cent = data[:k, :]
    nk = np.ones(k)

    for i, x in enumerate(data[k:, :]):
        x = x.reshape(1, mX)
        l = spdist.cdist(x, cent, metric=metric).argmin()

        nk[l] += 1
        cent[l] = cent[l] + (1/nk[l]) * (x - cent[l])

    labels = spdist.cdist(data, cent).argmin(axis=1)

    return cent, labels




def main():
    # Lloyd's algorithm
    centroids, _ = kmeans(data, k)
    idx, _ = vq(data, centroids)
    show_plotted_cluster(data, idx, centroids, "Lloyd's algorithm",k)

    # Hartigan's algorithm
    centroids, idx = kmeans_hartigans(data, k)
    show_plotted_cluster(data, idx, centroids, "Hartigan's algorithm",k)

    # MacQueen's algorithm
    centroids, idx = kmeans_macqueen(data, k)
    show_plotted_cluster(data, idx, centroids, "MacQueen's algorithm",k)

    return


#main()
