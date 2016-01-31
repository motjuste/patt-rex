import numpy as np
import scipy.linalg as la
import scipy.spatial as ss
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pattrex.plotting_mpl as plt_rex

def SimilarityMatrix(data, beta):
    SimMat = np.zeros(shape=(data.shape[1],data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            SimMat[i][j]=np.exp(-beta*ssd.euclidean(data[:,i],data[:,j])**2)
    return SimMat
   
def DiagonalMatrix(SimMat):
    DiaMat = np.zeros(shape=(SimMat.shape[0],SimMat.shape[1]))
    for i in range(SimMat.shape[0]):
        for j in range(SimMat.shape[1]):
            if i == j:
                DiaMat[i][j]=np.sum(SimMat[i,j:SimMat.shape[1]])
    return DiaMat

def LaplacianMatrix(data, beta):
    SimMat = SimilarityMatrix(data, beta)
    DiaMat = DiagonalMatrix(SimMat)
    LapMat = np.subtract(DiaMat, SimMat)
    return LapMat

def SpectralClustering(data, beta):
    LMat = LaplacianMatrix(data, beta)
    l, ur = la.eig(LMat, right=True)
    index=np.argpartition(l,2)
    u_idx_pos = np.where(np.sign(ur[:,index[1]])==1)
    u_idx_neg = np.where(np.sign(ur[:,index[1]])==-1)
    result=[]
    result.append(ur)
    result.append(index)
    result.append(u_idx_pos)
    result.append(u_idx_neg)
    return result

def plot(data,ur,index,u_idx_pos,u_idx_neg):

    fig = plt.figure(figsize=(18, 6))

    #Plot the second smallest eigen vetor u2    
    plt.subplot(131)
    x = u_idx_neg[0]
    y = ur[:,index[1]][u_idx_neg[0]]
    plt.ylim(-0.01,0.01)
    plt.scatter(x, y, color='b')
    x = u_idx_pos[0]
    y = ur[:,index[1]][u_idx_pos[0]]
    plt.ylim(-0.01,0.01)
    plt.scatter(x, y, color='r')
    #Plot the stubburn outlier
    x = np.arange(65,66,1)
    y = ur[:,index[1]][65]
    plt.scatter(x,y, color='g')

    #Plot the horizaontal line y=0, which we use to cluster
    plt.axhline(y=.0, xmin=-20, xmax=120, linewidth=1, color = 'k')

    #Plot the second smallest eigen vetor u2    
    plt.subplot(132)
    x = u_idx_neg[0]
    y = ur[:,index[1]][u_idx_neg[0]]
    plt.scatter(x, y, color='b')
    x = u_idx_pos[0]
    y = ur[:,index[1]][u_idx_pos[0]]
    plt.scatter(x, y, color='r')
    #Plot the stubburn outlier
    x = np.arange(65,66,1)
    y = ur[:,index[1]][65]
    plt.scatter(x,y, color='g')

    #Plot the horizaontal line y=0, which we use to cluster
    plt.axhline(y=.0, xmin=-20, xmax=120, linewidth=1, color = 'k')
    
    #Plot the cluster result
    axs2 = fig.add_subplot(133)

    plt_rex.plot2d(data[:,u_idx_neg[0]], colwise_data=True, hatch='bo', 
                  show=False, axs=axs2, set_aspect_equal=False, plotlabel="Neg")
    plt_rex.plot2d(data[:,u_idx_pos[0]], colwise_data=True, hatch='ro', 
                  show=False, axs=axs2, set_aspect_equal=False, plotlabel="Pos")