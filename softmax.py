"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	#enter = False
	#if isinstance(x,np.ndarray):
	#	if x.shape[0] == 1:
	#		enter = True
	#if isinstance(x,list): #Is a list? Each element is an element | enter=True	
	#	result = []
	#	denominator = sum(math.exp(x))
	#	for i in range(size(x)):
	#		result.append( math.exp(x(i))/denominator )
		#enter = False
	#else:
		# isinstance(x,np.ndarray) ? A matrix were each column represents a sample
		# x.shape[0] -> each item x.shape[1] -> each element of the item
	result = np.exp(x)
	#print len(result.shape)
	if len(result.shape) == 1:
		result = result / np.sum(result) # -> With just this line the function would have worked OK!
	else:
		nitem = result.shape[0]
		nelem = result.shape[1]
		#print nitem, nelem
		for i in range(nelem):
			denominator = np.sum ( result[:,i] )
			#print denominator
			for j in range(nitem):
				result[j,i] = result[j,i] / denominator 

	return result # TODO: Compute and return softmax(x)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
