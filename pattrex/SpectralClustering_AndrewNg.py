import numpy as np
from scipy.cluster.vq import kmeans, vq
import scipy.linalg as la
from numpy import genfromtxt
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

def AffinityMatrix(data, sigma):
    AffMat = np.zeros(shape=(data.shape[1],data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                AffMat[i][j]=np.exp(-ssd.euclidean(data[:,i],data[:,j])**2/(2*(sigma**2)))
    return AffMat
   
def DiagonalMatrix(A):
    D = np.sum(A, axis=1) * np.eye(A.shape[0])
    return D

def LaplacianMatrix(D, A):
    L = np.dot(np.linalg.inv(np.power(D, 0.5)), np.dot(A, np.linalg.inv(np.power(D, 0.5))))
    return L

def plot(dataProcessed, idx, centroids, k, dataOri):
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.scatter(dataProcessed[:, 0], dataProcessed[:, 1], marker='o', c=idx, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c=range(k))
    
    #Plot the cluster result
    plt.subplot(122)
    plt.scatter(dataOri[0,:], dataOri[1,:], marker='o', c=idx, alpha=0.5)
    
def SpectralClustering(data, k, sigma):
    LM=LaplacianMatrix(DiagonalMatrix(AffinityMatrix(data,sigma)),AffinityMatrix(data,sigma))

    l, ur = la.eig(LM, right=True)
    index=np.argsort(l)

    X = ur[:,index[index.shape[0]-k:index.shape[0]]]
    D = np.sqrt((np.sum((np.square(X)),axis=1)))
    D = np.vstack((D,D)).T
    Y = np.divide(X,D)
    centroids, distortion = kmeans(Y, k)
    idx, _ = vq(Y, centroids)

    result = []
    result.append(idx)
    result.append(centroids)
    result.append(distortion)
    result.append(Y)
    
    return result    

def demo1(my_data,k,start, end, step):
    if(np.ceil(np.abs(end-start)/step)>15):
        print("Too much iteration")
        return 
    else:
        for i in np.arange(start, end, step):
            idx, centroids, distortion, Y = SpectralClustering(my_data, k, i)
            plot(Y,idx,centroids,k,my_data)
