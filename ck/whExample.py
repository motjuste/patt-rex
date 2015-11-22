
import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

def FitNormalDistribution(X, filename=None):
    
    #Set up the plot
    fig = plt.figure()
    axs = fig.add_subplot(111)

    xmin = X.min()
    xmax = X.max()

    #Set the boundary
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(0,0.06)    #TODO:How to generate this Ymax
    
    # plot the data 
    axs.plot(X, len(X)*[0],'ro', label='data')
    
    #Sort X
    SortedX = np.sort(X)
    
    meanWeight = np.mean(SortedX)
    print meanWeight

    stdWeight = np.std(SortedX)
    print stdWeight

    #fit a normal distibution
    meanWeight, stdWeight = norm.fit(SortedX)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, meanWeight, stdWeight)
    axs.plot(x,p,'k',linewidth=2)
 
        # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    #axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    print "X[0,:]"
    print X[0,:]
    print "\nX[1,:]"
    print X[1,:]
    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
# dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
#    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
#    ws = np.array([d[0] for d in data])
#    hs = np.array([d[1] for d in data])
#    gs = np.array([d[2] for d in data]) 


    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)
    # read height and weight data into 2D array (i.e. into a matrix)
    print data

    X = data[:,0:2].astype(np.float)
    ListToDelete = []
    # read gender data into 1D array (i.e. into a vector)
    for i in range(len(X)):
        for j in range(len(X[i,])):
            if X[i,j] < 0:
                ListToDelete.append(i)
              
    X = np.delete(X,ListToDelete,0)
        

    y = data[:,2]
    print y
    # let's transpose the data matrix 
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X, 'plotWH.pdf')

    # next, let's plot height vs. weight 
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0,:])
    h = np.copy(X[1,:])
    
    # second, create new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # third, plot this new representation of the data
    plotData2D(Z, 'plotHW.pdf')
    
    #task 1.2
    FitNormalDistribution(X[1,:],'plotFitNormalDistWH.pdf')
    