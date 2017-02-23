from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def waveform(data, fs = 1500):
	'''data is 1d array of amplitudes for a given time. fs is sampling frequency of the signal'''
	data_len = len(data)
	time_array = np.arange(data_len)/fs
	return (time_array,data)
	
	
def times(data, fs = 1500):
	'''data is 1d array of amplitudes for a given time. fs is sampling frequency of the signal'''
	data_len = len(data)
	time_array = np.arange(data_len)/fs
	return (time_array)
	
if __name__ == '__main__':

	data = np.loadtxt('Team 17/12V motor x axis ten minutes.csv', delimiter=',', comments='#')[:,1][:5000]
	times = waveform(data)[0]
	print times
	print data
	plt.plot(times, data)
	plt.show()