import numpy as np
import numpy.linalg as la

#1.Compute Nearest neighbor
#1.1 Naïve way: O(n^2) ( n*(n-1) ) for all
#https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors?channel=doi&linkId=563f695508ae45b5d28d2ea9&showFulltext=true

def KNN(X, q, k, method):
    if(k<1):
        print("K should be integer greater than or equal to 1.")
        return
    elif(k==1):
        if(method==0):
            return nearest_neighbor_method0(X,q)
        elif(method==1):
            return nearest_neighbot_method1(X,q)
        elif(method==2):
            return nearest_neighbot_method2(X,q)
        elif(method==3):
            return nearest_neighbot_method3(X,q)
        else:
            print("Support 4 method, method = i to indicate.")
            return
    else:
        if(method==0):
            return k_nearest_neighbors(X,q)
        elif(method==1):
            return k_nearest_neighbors_par(X,q)
        elif(method==2):
            return k_nearest_neighbors_smallk(X,q)
        else:
            print("Support 3 method, method = i to indicate.")
            return
            

def nearest_neighbor_method0(X,q):
    m, n = X.shape
    sqr = np.square(np.subtract(X.T,q))# (X-q)^2
    _sum = np.add(sqr[:,0],sqr[:,1]) #sum up the x and y
    return np.argmin(_sum) # retun the argmin

def nearest_neighbor_method1(X, q):
    m, n = X.shape
    minindx = 0
    mindist = np.inf
    for i in range(n):
        dist = la.norm(X[:,i] - q)
        if dist <= mindist:
            mindist = dist
            minindx = i
    return minindx

def nearest_neighbor_method2(X, q):
    m, n = X.shape
    return np.argmin(np.sum((X-q.reshape(m,1))**2, axis=0))

def nearest_neighbor_method3(X, q):
    X = X.T
    return np.argmin(np.sum((X - q)**2, axis=1))

def k_nearest_neighbors(X, q, k):
    X = X.T
    sorted_inds = np.argsort(np.sum((X - q)**2, axis=1))
    return sorted_inds[:k]

def k_nearest_neighbors_par(X, q, k):
    X=X.T
    sorted_inds = np.argpartition(np.sum((X - q)**2, axis=1), k-1)
    return sorted_inds[:k]

def k_nearest_neighbors_smallk(X, q, k):
    inds=nearest_neighbor_method3(X, q)
    a_inds = np.array(inds)
    X=np.delete(X, inds, axis=1)
    for i in range(k-1):
        inds=nearest_neighbor_method2(X, q)
        a_inds=np.append(a_inds,inds)#remember to assign a pointer to new array. the return value is a pointer
        if i!=k-1:
            X=np.delete(X, inds, axis=1)
    return a_inds

