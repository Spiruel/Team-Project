import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
import Audio_waveform as am
import pandas


def butter_lowpass(cutoff, fs, order=5):
	'''cutoff: cutoff frequency of filter
	fs: sampling frequency
	order: order of the filter
	returns b and a which characterise the filter'''
	nyq = 0.5 * fs 
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	'''applies filter to the data and returns y, which is filtered data'''
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y
    

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    
def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

def rolling_average(data, window_size):
	dataseries = pandas.Series(data)
	rolling_data = dataseries.rolling(window=window_size,center=False).mean()
	rolling_data = np.array(rolling_data)

	return rolling_data

def rolling_std(data, window_size):
	dataseries = pandas.Series(data)
	rolling_std = dataseries.rolling(window=window_size,center=False).std()
	rolling_std = np.array(rolling_std)

	return rolling_std



if __name__ == '__main__':
	
	# Filter requirements.
	order = 6
	fs = 3000      # sample rate, Hz
	cutoff = 50  # desired cutoff frequency of the filter, Hz
	
	data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')[:,0]

	#y = butter_lowpass_filter(data, cutoff, fs, order)
	
	times = am.times(data, fs)
	
	average = rolling_average(data,50)

	
	plt.plot(times,data)
	plt.plot(times,average)
	plt.show()