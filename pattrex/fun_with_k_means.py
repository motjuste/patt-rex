"""
@aksakalli

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

#DATA_PATH = "../data/data-clustering-1.csv"

#data = np.genfromtxt(DATA_PATH, delimiter=",")
#data = np.transpose(data)

#k = 3


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
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=idx, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(k))
    plt.title(title)
    plt.show()
    return


def save_plotted_cluster(data, idx, centroids, title, file_prefix, iteration, k):
    plt.cla()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=idx, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(k))
    plt.title(title)
    plt.savefig('img/' + file_prefix + '-' + str(iteration).zfill(5) + '.png')
    return


# importan! comment out save_ploted_cluster calls for performance test

def kmeans_Lloyd(data, k, centroids):
    
    ## Important!! the input parameter is different to others, this requires a initial centroid
    
    # Set t=0 and initialize centroids
    iteration = 0
    maxIteration = 200
    prefix = 'Lloyd'
    title = "Lloyd's algorithm, update: {}"

    colors = np.zeros((len(data), 3))
    idx = np.zeros(len(data))
    # initial plot
    save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration, k)

    # repeat until convergence
    converged = False
    while not converged:

        idxPrev = np.copy(idx)
        #update all clusters
        for i in range(len(data)):
            candidate_class = 0
            candidate_distance = np.inf
            for j in range(k):
                distance = np.linalg.norm(centroids[j] - data[i, :])
                if distance < candidate_distance:
                    candidate_class = j
                    candidate_distance = distance
            idx[i] = candidate_class

        show_plotted_cluster(data, idx, centroids, "Lloyd's algorithm",k)
        # update all cluster centroid
        centroidsPrev = centroids
        centroids = compute_centroids(data, idx, k)

        # increase iteration counter
        iteration=iteration+1
        # save the current progress
        save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration, k)

        if(iteration>=maxIteration):
            converged = True
            convCondition = 1
        if(np.array_equal(idxPrev,idx)):
            converged = True
            convCondition = 2
        if(np.allclose(centroids,centroidsPrev)):
            converged = True
            convCondition = 3

    return centroids, idx, convCondition

def kmeans_hartigans(data, k):
    # plotting purpose
    iteration = 0
    prefix = 'hartigan'
    title = "Hartigan's algorithm, update: {}"

    # randomly assign all data to a cluster
    idx = np.random.randint(k, size=len(data))
    # compute the initial centroids
    centroids = compute_centroids(data, idx, k)

    # initial plot
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
                save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration, k)

            idx[i] = candidate_class


        if converged:
            break

    return centroids, idx


def kmeans_macqueen(data, k):
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
    save_plotted_cluster(data, idx, centroids, title.format(iteration), prefix, iteration,k)

    return centroids, idx


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
