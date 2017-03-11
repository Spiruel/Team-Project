import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
import Audio_waveform as am
import pandas
import Fourier

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
	rolling_data = dataseries.rolling(window=window_size,center=True).mean()
	rolling_data = np.array(rolling_data)

	return rolling_data

def rolling_std(data, window_size):
	dataseries = pandas.Series(data)
	rolling_std = dataseries.rolling(window=window_size,center=True).std()
	rolling_std = np.array(rolling_std)

	return rolling_std



if __name__ == '__main__':
	
	# Filter requirements.
	order = 6
	fs = 3000      # sample rate, Hz
	cutoff = 50  # desired cutoff frequency of the filter, Hz

	
	data = np.loadtxt('data/large_0V.csv', delimiter=',', comments='#',skiprows=1)[:,1][0:5000]
	freqs_and_amps = Fourier.fourier(data, fs)
	freqs = freqs_and_amps[0]
	amplitudes = freqs_and_amps[1]
	other_amplitudes = Fourier.fourier_transform(data)
	smoothed_amps = movingaverage(amplitudes,20)

	f = plt.figure(figsize=(8,4))
	ax1 = plt.subplot(211)

	plt.plot(freqs,amplitudes/1e-7)
	plt.plot(freqs, smoothed_amps/1e-7)
	plt.xlim(0,1500)

	ax2 = plt.subplot(212)

	plt.semilogy(freqs,amplitudes)
	plt.semilogy(freqs, smoothed_amps)
	plt.xlim(0,1500)

	ax1.set_ylabel('$|FFT|^2$ \n / $10^{-7}$ V$^2$$\;$Hz$^{-1}$', fontsize=14)
	ax2.set_ylabel('$|FFT|^2$ \n / V$^2$$\;$Hz$^{-1}$' , fontsize=14)
	ax2.set_xlabel('Frequency / Hz', fontsize=14)

	f.subplots_adjust(hspace=0)
	plt.style.use('seaborn-white')

	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

	#plt.savefig('figures/freq_moving_average.pdf', dpi=300, transparent=True, bbox_inches='tight')
	#plt.savefig('figures/freq_moving_average.png', dpi=300, transparent=True, bbox_inches='tight')
	plt.show()

	#y = butter_lowpass_filter(data, cutoff, fs, order)
	




	times = am.times(data, fs)
	
	average = movingaverage(data,20)

	plt.figure(figsize=(10,2))
	
	plt.plot(times,data)
	plt.plot(times,average)

	plt.xlabel('Time / s', fontsize=14)
	plt.ylabel('Amplitude / V', fontsize=14)

	plt.xlim(0,max(times))


	plt.style.use('seaborn-white')
	#plt.savefig('figures/moving_average.pdf', dpi=300, transparent=True, bbox_inches='tight')
	#plt.savefig('figures/moving_average.png', dpi=300, transparent=True, bbox_inches='tight')
	plt.show()
