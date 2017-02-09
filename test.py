import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.optimize import leastsq
import operator
import Filters


data = np.loadtxt('TracerCo Project/Team-Project/Team 17/sin wave test.csv', delimiter=',', comments='#')
time, amplitude = data[:,0], data[:,1]

sample_rate = 1500.0
sample_interval = 1.0/1500.0
n = len(time)
k = np.arange(n)
T = n / sample_rate
frequency = k / T
frequency = frequency[range(n/2)]

    
Y = fft(amplitude) / n
Y = abs(Y[range(n/2)])
#    Y = 20*np.log10(Y)


Y[0] = 0



maxindex_Y, maxvalue_Y = max(enumerate(Y), key=operator.itemgetter(1))
print maxvalue_Y, frequency[maxindex_Y]

plt.plot(frequency, Y) 
plt.scatter(frequency[maxindex_Y], maxvalue_Y, c = 'r', alpha = 0.5)
#plt.xlim(299, 301)
#plt.ylim(0, 0.0048)

plt.show()