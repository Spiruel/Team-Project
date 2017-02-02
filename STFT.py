from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


data = np.loadtxt('Team 17/12V motor x axis.csv', delimiter=',',skiprows = 3)[:,1]
fs = 1500 #sampling frequency

# data = a numpy array containing the signal to be processed
# fs = a scalar which is the sampling frequency of the data
def STFT(data, fs = 1500, clip = False):
	'''data is 1d array of amplitudes for a given time. fs is sampling frequency of the signal'''

	fft_size = 500 #number of samples used for window
	overlap_fac = 0.5 #overlap factor
 
	hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
	pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
	total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
	t_max = len(data) / np.float32(fs)
 
	window = np.hanning(fft_size)  # our half cosine window
	inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
 
	proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
	result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result
 
	for i in xrange(total_segments):                      # for each segment
	    current_hop = hop_size * i                        # figure out the current segment offset
	    segment = proc[current_hop:current_hop+fft_size]  # get the current segment
	    windowed = segment * window                       # multiply by the half cosine function
	    padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
	    spectrum = np.fft.fft(padded) #/ fft_size          # take the Fourier Transform and scale by the number of samples
	    autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
	    #print np.fft.fftfreq(fft_size, d = np.float32(1/fs)) 
	    result[i, :] = autopower[:fft_size]               # append to the results array
 
 	clip_freq = 0
 	if clip:
 		clip_freq = clip_freq + 30
 		result = result[:,clip_freq:]
 	
	result = 20*np.log10(result)          # scale to db
	#result = np.clip(result, -40, 200)  
	
	max_freq = fs/2 #divided by two as the sample window was extended with zeros.
	max_time = t_max
	
	Hz_clip = clip_freq/fft_size * max_freq 
	print Hz_clip

	return (result, max_freq, max_time, Hz_clip)

if __name__ == '__main__':

	result = STFT(data,fs,True)
	
	amplitudes = result[0]
	max_freq = result[1]
	max_time = result[2]
	clip_freq = result[3]
	
	ax = plt.subplot(111)
	
	img = ax.imshow(amplitudes, origin='lower', cmap='jet', interpolation='nearest', aspect='auto', extent = [clip_freq,max_freq,0,max_time])
	ax.set_xlabel('Frequency / Hz')
	ax.set_ylabel('Time / s')
	
	#ax.set_xticks(freq_ax)
	
	plt.colorbar(img)
	plt.show()