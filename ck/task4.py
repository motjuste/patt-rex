
import numpy as np
import scipy as sp
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    numSamplePoints = 100;  
    arrayX = np.linspace(0.0,1.0,num = numSamplePoints)
    arrayY = np.zeros(numSamplePoints)
    for i in range(numSamplePoints):
    	print i
        arrayY[i]=1 -2*math.sqrt(arrayX[i])+arrayX[i]
        print arrayX[i]
        print arrayY[i]

    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_xlim(-1.0,1.0)
    axs.set_ylim(-1.0,1.0)

    #plot using symmetric
    #1. First Quadrant
    plt.plot(arrayX, arrayY)
    #2. Second Quadrant
    plt.plot(-arrayX, arrayY)
    #3. Third Quadrant
    plt.plot(-arrayX, -arrayY)
    #4. Fourth Quadrant
    plt.plot(arrayX, -arrayY)
    plt.plot(arrayX,-arrayY)
    plt.show()
    plt.close()

    print "Hello"