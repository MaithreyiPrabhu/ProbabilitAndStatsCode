import numpy as np
import pylab
import scipy.stats as stats

#N(0,1)
standard_normal = np.random.normal(loc=0,scale=1,size=1000)
for i in range(0,101):
    print(i,np.percentile(standard_normal,i))
measurements = np.random.normal(loc=20,scale=5,size=100)
stats.probplot(measurements,dist="norm",plot=pylab)
pylab.show()