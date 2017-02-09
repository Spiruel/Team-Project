from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
    
def find_peaks(freq_amps, fs=3000):
	'''Takes inputs:
	freq_amps -- array of amplitudes corresponding to certain frequencies in the fourier plot
	fs -- sampling frequency
	
	returns:
	peak_freqs -- array containing the frequency of the peaks found
	peak_amps -- array containing the corresponding amplitudes of the peaks'''
	length = len(freq_amps)
	peak_width_range = np.arange(length/1000,length/10, length/1000)
	peak_indices = np.array(signal.find_peaks_cwt(freq_amps, peak_width_range, min_snr = 3))
	
	peak_amps = np.array([])
	for i in range(len(peak_indices)):
		index = peak_indices[i]
		peak_amps = np.append(peak_amps, freq_amps[index])
	
	peak_freqs = peak_indices/length*fs/2
	
	return (peak_freqs, peak_amps)
	
if __name__ == '__main__':
	
	data = np.loadtxt('Team 17/12V motor x axis ten minutes.csv', delimiter=',', comments='#')[:,1][:5000]
	
	fs = 1500
	
	import Fourier
	
	fourier = Fourier.fourier(data,fs)
	
	freqs = fourier[0]
	freq_amps = fourier[1]
	freq_amps = 20*np.log10(freq_amps)
	
	peaks = find_peaks(freq_amps, fs)
	
	peak_freqs = peaks[0]
	peak_amps = peaks[1]
	
	plt.plot(freqs,freq_amps)
	plt.plot(peak_freqs,peak_amps, 'ro', markersize = 10)
	plt.show()