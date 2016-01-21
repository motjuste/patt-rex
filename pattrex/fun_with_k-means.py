import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

DATA_PATH = "../data/data-clustering-1.csv"

data = np.genfromtxt(DATA_PATH, delimiter=",")
data = np.transpose(data)

centroids, _ = kmeans(data, 3)
idx, _ = vq(data, centroids)

plt.scatter(data[:, 0], data[:, 1], marker='o', c=idx, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(3))
plt.show()
