import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import exponweib


if __name__ == "__main__":
	#Read data
	data = np.loadtxt("myspace.csv",dtype=np.object,delimiter=',')
	#Select the 2nd Column
	prevH = data[:,1:2].astype(np.float)
	#Remove leading zeros
	ListToDelete = []

	for i in range(len(prevH)):
		if prevH[i] <= 0:
			ListToDelete.append(i)
  
	H = np.delete(prevH,ListToDelete,0)

	print H
	#X=[1,2,3,...,n]
	#X = np.arange(1,len(H)+1)	
	#plt.plot(H, exponweib.pdf(H, *exponweib.fit(H, 1, 1, scale=10, loc=0)))
	#Set up the plot
