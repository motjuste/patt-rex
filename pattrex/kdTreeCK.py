import numpy as np
import pattrex.plotting_mpl as plt_rex
import matplotlib.pyplot as plt

class Node:
    def __init__(self, split, left, right, sliceDimension, bound):
        self.split = split
        self.left = left
        self.right = right
        self.sliceDimension = sliceDimension
        self.bound = bound #NOTE: The bounding box here is not determined by the max and min of this part of data, it is deciede by the whole data
               
def KDTree(X, depth, dim, splt, bound=None):
    print("Depth:",depth)

    n, k = X.shape
    print("The number of points is:", n, "The dimension is:", k)
    if(n==0):
        print("n:",n,"###Stopped by n, depth is ", depth)
        return None
    
    if(depth==10):#safety valve, remove if everything is checked.
        print(depth,"!!!Stopped by depth valve. n is ",n)
        return None
    
    #Dimension selection
    if(dim==0): #Alternate betwenn x and y
        slcDim = depth % k  #refactor d as slcDim sliceDimension
    elif(dim==1): #Split along dimension of higer variance
        slcDim = np.argmax( np.var(X[:,:], axis=0) )        
    else:
        print("dim should be either 1:Alternate or 2:Highest variance")
    print("SliceDimension: ",slcDim)
    
    #Split point selection
    if(splt==0):#Median
        j = np.median(X[:,slcDim])
        if (bound==None):#bound==None
            bound = np.stack((np.amax(X,axis=0),np.amin(X,axis=0)),axis=0)
        i = np.average(np.stack((np.amax(X,axis=0),np.amin(X,axis=0)),axis=0),axis=0)
        i[slcDim]=j
        #Design Choice: Since we calculate the median by only one axis
        #if the number of data is even, we would get the result as the sum of the very middle two points 
        #on the desinated axis, but simply interpolate the unused axis is meaningless, I then store nothing but the 
        #desinated axis. But in order to get a clean result, we use the middle point method on those axises.      
    elif(splt==1):#Midpoint, middle of the bounding box
        if (bound==None):#bound==None
            bound = np.stack((np.amax(X,axis=0),np.amin(X,axis=0)),axis=0)
        i=np.average(np.stack((np.amax(X,axis=0),np.amin(X,axis=0)),axis=0),axis=0)
    elif(splt==2):#sliding-Midpoint #OPTIONAL
        i=-1
    else:
        print("splt should be either 1:Midpoint or 2:Median")
    psu_lb=np.copy(bound)
    psu_rb=np.copy(bound)
    psu_lb[0,slcDim]=i[slcDim]
    psu_rb[1,slcDim]=i[slcDim]
    print("psu_lb, After sub",psu_lb)
    print("psu_rb, After sub",psu_rb)

    #Split the array
    print(i[slcDim])
    if(n%2==0):
        M = X[:,slcDim] <= i[slcDim]
        left, right = X[M], X[~M]
    elif(n%2==1):
        L = X[:,slcDim] < i[slcDim]
        R = X[:,slcDim] > i[slcDim]
        left, right = X[L], X[R]
    
    return Node(i, 
                KDTree(left, depth+1, dim, splt, psu_lb), 
                KDTree(right, depth+1, dim, splt, psu_rb),
                slcDim,
                bound
                )

def KDTreePlotBranch2D(Node, axs):

    if(Node==None):
        return
    
    if(axs==None):
        fig = plt.figure(figsize=(12, 12))
        axs = fig.add_subplot(221)
        
    axisMax = np.amax(Node.bound[:,~Node.sliceDimension])
    axisMin = np.amin(Node.bound[:,~Node.sliceDimension])
    midPoint = Node.split[Node.sliceDimension]
    print(Node.bound)
    print("max", axisMax,"min", axisMin, "mid",midPoint )
    # plt.plot([axisMin, axisMax], [midPoint, midPoint], color='b', linestyle='-', linewidth=1)

    if Node.sliceDimension==0:
        axs.plot([midPoint, midPoint], [axisMin, axisMax], color='b', linestyle='-', linewidth=1)
    elif Node.sliceDimension==1:
        axs.plot([axisMin, axisMax], [midPoint, midPoint], color='g', linestyle='-', linewidth=1)
    
    KDTreePlotBranch2D(Node.left,axs)
    KDTreePlotBranch2D(Node.right,axs)
    return

def KDTreeTraverse(Node):
    #global global_traverseCounter
    #if(global_traverseCounter>4096):
    #    print("Exit")
    #    return
    #else:
    #    global_traverseCounter=global_traverseCounter+1

    if(Node==None):
        return
    
    print(Node.bound)
    
    print("Go Left")
    KDTreeTraverse(Node.left)
    print("Go Right")
    KDTreeTraverse(Node.right)
    return



def PlotBaseAndScatter(x,y,axsList):
    X = np.vstack((x, y))  # only the measurements; data is col-wise
    xmin, ymin = X.min(axis=1)
    xmax, ymax = X.max(axis=1)

    xlim = [xmin, xmax]  # purely for looks
    ylim = [ymin, ymax]
    
    plt_rex.plot2d(X, colwise_data=True, hatch='ro', x_lim=xlim, 
                   y_lim=ylim, show=False, axs=axsList[0], set_aspect_equal=False, 
                   title="Alternative and Median")
    plt_rex.plot2d(X, colwise_data=True, hatch='ro', x_lim=xlim, 
                   y_lim=ylim, show=False, axs=axsList[1], set_aspect_equal=False, 
                   title="High Variance and Median")
    plt_rex.plot2d(X, colwise_data=True, hatch='ro', x_lim=xlim, 
                   y_lim=ylim, show=False, axs=axsList[2], set_aspect_equal=False, 
                   title="Alternative and MidPoint")
    plt_rex.plot2d(X, colwise_data=True, hatch='ro', x_lim=xlim, 
                   y_lim=ylim, show=False, axs=axsList[3], set_aspect_equal=False, 
                   title="High Variance and MidPoint")
    return

def KDTreePlot2D(x, y, TreeList):
# def KDTreePlot2D(Node1,Node2,Node3,Node4):
    
    #Four variants of KDTree
    fig = plt.figure(figsize=(12, 12))
    axsList=[]
    
    for i in range(TreeList.__len__()):
        axsList.append(fig.add_subplot(TreeList.__len__()/2 + TreeList.__len__()%2, 2 ,i+1))

    PlotBaseAndScatter(x,y, axsList)
    
    for i in range(TreeList.__len__()):
        KDTreePlotBranch2D(TreeList[i],axsList[i])
    
    return