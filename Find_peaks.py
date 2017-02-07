from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
    
def find_peaks(freq_amps, fs=3000):
	length = len(freq_amps)
	peak_width_range = np.arange(length/1000,length/10, length/1000)
	peak_indices = np.array(signal.find_peaks_cwt(freq_amps, peak_width_range, min_snr = 2))
	
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
	
	freqs = Fourier.fourier(data,fs)[0]
	freq_amps = Fourier.fourier(data, fs)[1]
	freq_amps = 20*np.log10(freq_amps)
	
	peak_freqs = find_peaks(freq_amps, fs)[0]
	peak_amps = find_peaks(freq_amps, fs)[1]
	
	plt.plot(freqs,freq_amps)
	plt.plot(peak_freqs,peak_amps, 'ro', markersize = 30)
	plt.show()