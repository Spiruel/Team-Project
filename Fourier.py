from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def fourier(data, fs = 3000):
	'''takes in amplitudes over time and sampling rate, and returns frequencies and their corresponding amplitudes'''
	n = len(data)
	spectrum = np.fft.fft(data)[range(int(n/2))]
	freq_amps = np.abs(spectrum * np.conj(spectrum))
	freqs = np.arange(len(data))/n*fs
	freqs = freqs[range(int(n/2))]
	return (freqs, freq_amps)
	
if __name__ == '__main__':
	data = np.loadtxt('Team 17/12V motor x axis ten minutes.csv', delimiter=',', comments='#')[:,1][:5000]
	fs = 1500
	freqs = fourier(data, fs)[0]
	freq_amps = fourier(data, fs)[1]
	freq_amps = 20*np.log10(freq_amps)
	#plt.xlim(,fs)
	#plt.ylim(0,max(freq_amps))
	
	print len(freq_amps), "@@@@@@@@@"
	
	peak_freqs = np.array(signal.find_peaks_cwt(freq_amps, np.arange(len(freq_amps)/1000,len(freq_amps)/10,len(freq_amps)/1000), min_snr = 2))#/len(freq_amps)*fs/2
	print peak_freqs
	
	
	
	plt.vlines(x=peak_freqs, ymin=-50, ymax=50, color='red', lw=1, linestyle='-')
	plt.plot(freqs,freq_amps)
	plt.show()