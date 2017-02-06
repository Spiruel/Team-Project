from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def fourier(data, fs = 1500):
	n = len(data)
	spectrum = np.fft.fft(data)[range(int(n/2))]
	freq_amps = np.abs(spectrum * np.conj(spectrum))
	freqs = np.arange(len(data))/n*fs
	freqs = freqs[range(int(n/2))]
	return (freqs, freq_amps)
	
if __name__ == '__main__':
	data = np.loadtxt('Team 17/sin wave test.csv', delimiter=',', comments='#')[:,1]
	fs = 1500
	freqs = fourier(data, fs)[0]
	freq_amps = fourier(data, fs)[1]
	freq_amps = 20*np.log10(freq_amps)
	#plt.xlim(,fs)
	#plt.ylim(0,max(freq_amps))
	plt.plot(freqs,freq_amps)
	plt.show()