from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import signal
from scipy.fftpack import fft
from scipy.optimize import leastsq
import Filters
import operator
from scipy.signal import butter, lfilter
import Find_peaks
import peakutils
import lmfit
from lmfit.models import GaussianModel, ConstantModel, LinearModel, LorentzianModel, Model



def conv_to_freq(data, indices):
	'''
	Takes in data set and indices of peaks in frequency space and converts the indices to frequency
	'''
	sample_rate = 3000
	n=len(data) #this is the only reason why data is a variable, to get its length
	T = n / sample_rate
	frequencies = indices/n * sample_rate
	return frequencies


def conv_to_samples(data, frequencies):

	sample_rate = 3000
	n = len(data)
	indices = frequencies/sample_rate * n
	return indices


def find_peaks(data):
	frequencies, sample_rate, amplitudes = params(data)
	peak_indices = peakutils.indexes(amplitudes, thres=0.4, min_dist=len(data)/20)
	peak_freqs =  conv_to_freq(data, peak_indices)

	peak_amplitudes = amplitudes[peak_indices]

	return (peak_indices, peak_freqs, peak_amplitudes)


def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))
    #another equation for it could be (amp) * np.exp(-(x-cen)**2 /(2*wid**2)) but the current one we are using makes the total area 
    #under the graph equal to amp, which is what we want. We want to monitor the change in the total amplitude of the peak, summed 
    #over all frequencies.

def lorentzian(x, amp, cen, wid):
	"1-d lorentzian: lorentzian(x, amp, cen, wid)"
	return amp/np.pi * wid/((x-cen)**2 + wid**2)

def line(x, slope, intercept):
    "line"
    return slope * x + intercept


	
'''
Function to return a set of parameters used throughout the rest of the code
''' 
def params(data):
	amplitude = data#[:,0]
	
	sample_rate = 3000.0
	n = len(amplitude)
	k = np.arange(n)
	T = n / sample_rate
	frequency = k / T
	frequency = frequency[range(np.int(n/2))]
	lowcut = 100
	Y = fourier_transform(amplitude, sample_rate, lowcut)
	Y_av = Filters.movingaverage(Y, 30) 
	return frequency, sample_rate, Y_av

'''
Functions to return a highpassed data set to reduce noise from lower frequencies
'''
def butter_highpass(lowcut, sample_rate, order=5):
	nyq = 0.5 * sample_rate
	low = lowcut / nyq
	b, a = butter(order, low, btype ='high')
	return b, a
	
def butter_highpass_filter(data, lowcut, sample_rate, order=5):
	b, a = butter_highpass(lowcut, sample_rate, order)
	y = lfilter(b, a, data)
	return y

'''
Function to return the intensity of the wave at certain amplitudes in 
Fourier space
'''
def fourier_transform(amplitude, sample_rate, lowcut):
	dat = butter_highpass_filter(amplitude, lowcut, sample_rate)
	Y = fft(dat) / len(amplitude)
	Y = abs(Y[range(np.int(len(amplitude)/2))])
	return Y
	
'''
Function used to optimise the Lorentzian parameters
'''
def residuals(p, Y_av, frequency):
	err = Y_av - lorentzian(frequency, p)
	return err
	
'''
Function to return the background noise of the data
'''
def background_subtraction(Y_av, frequency, sample_rate):
	ind_bg_low = (frequency > min(frequency)) & (frequency < frequency[peak_finder(frequency, Y_av, sample_rate)[2]]-20) #defining background
	ind_bg_high = (frequency > frequency[peak_finder(frequency, Y_av, sample_rate)[2]]+20) & (frequency < max(frequency))
	frequency_bg = np.concatenate((frequency[ind_bg_low], frequency[ind_bg_high]))
	Y_bg = np.concatenate((Y_av[ind_bg_low], Y_av[ind_bg_high]))
	m, c = np.polyfit(frequency_bg, Y_bg, 1)
	background = m * frequency + c
	Y_bg_corr = Y_av - background
	return background, Y_bg_corr

'''
Performs a Lorentzian fit using optimized parameters using least squares
'''
def optimization(frequency, p, Y_av, sample_rate):
	pbest = leastsq(residuals, p, args = (Y_av, frequency), full_output = 1)
	best_parameters = pbest[0]
	p_new = (best_parameters[0], best_parameters[1], peak_finder(frequency, Y_av, sample_rate)[1] - background_subtraction(Y_av, frequency, sample_rate)[0])
	fit = lorentzian(frequency, p_new)
	return fit

def fit_peaks(data):
	frequencies, sample_rate, amplitudes = params(data)

	peak_indices, peak_freqs, peak_amplitudes = find_peaks(data)

	peak_index = peak_indices[0]
	peak_freq = peak_freqs[0]
	peak_amp = peak_amplitudes[0]

	min_freq, max_freq = peak_freq-90, peak_freq+90

	mod = Model(lorentzian) + Model(line)
	pars  = mod.make_params( amp=peak_amp, cen=peak_freq, wid=5, slope=0, intercept=0)

	amp_range = amplitudes[conv_to_samples(data,min_freq):conv_to_samples(data,max_freq)]
	freq_range = frequencies[conv_to_samples(data,min_freq):conv_to_samples(data,max_freq)]

	out = mod.fit(amp_range, pars, x=freq_range)

	amplitude, amplitude_err = out.params['amp'].value, out.params['amp'].stderr
	centre, centre_err = out.params['cen'].value, out.params['cen'].stderr
	sigma, sigma_err = out.params['wid'].value, out.params['wid'].stderr
	slope, slope_err = out.params['slope'].value, out.params['slope'].stderr
	c, c_err = out.params['intercept'].value, out.params['intercept'].stderr

	all_parameters = np.array([amplitude,centre,sigma,slope,c])
	all_parameters_err = np.array([amplitude_err, centre_err, sigma_err, slope_err, c_err])

	if len(peak_indices)>1:

		for i in range(len(peak_indices)-1):
			peak_index = peak_indices[i+1] #plus one as you've already done i=0 outside of the for loop
			peak_freq = peak_freqs[i+1]
			peak_amp = peak_amplitudes[i+1]

			min_freq, max_freq = peak_freq-90, peak_freq+90

			mod = Model(gaussian) + Model(line)
			pars  = mod.make_params( amp=peak_amp, cen=peak_freq, wid=5, slope=0, intercept=0)

			amp_range = amplitudes[conv_to_samples(data,min_freq):conv_to_samples(data,max_freq)]
			freq_range = frequencies[conv_to_samples(data,min_freq):conv_to_samples(data,max_freq)]

			out = mod.fit(amp_range, pars, x=freq_range)

			amplitude, amplitude_err = out.params['amp'].value, out.params['amp'].stderr
			centre, centre_err = out.params['cen'].value, out.params['cen'].stderr
			sigma, sigma_err = out.params['wid'].value, out.params['wid'].stderr
			slope, slope_err = out.params['slope'].value, out.params['slope'].stderr
			c, c_err = out.params['intercept'].value, out.params['intercept'].stderr

			parameters = np.array([amplitude,centre,sigma,slope,c])
			all_parameters = np.vstack((all_parameters,parameters))

			parameters_err = np.array([amplitude_err, centre_err, sigma_err, slope_err, c_err])
			all_parameters_err = np.vstack((all_parameters_err,parameters_err))


	return (all_parameters, all_parameters_err)

if __name__ == '__main__':
	data = np.loadtxt('data/12v_comparisontobaseline.csv', delimiter=',', comments='#')[:,0][5000:10000]
	frequencies, sample_rate, amplitudes = params(data)
	
	peak_indices, peak_freqs, peak_amplitudes = find_peaks(data)
	print peak_freqs
	

	plt.plot(params(data)[0], fourier_transform(data, 3000, 100), label = 'Raw data')
	plt.plot(params(data)[0], amplitudes, label = 'Smoothed data')
	#plt.plot(params(data)[0], optimization(params(data)[0], lorentz_params(data), params(data)[2], params(data)[1]) + background_subtraction(params(data)[2], params(data)[0], params(data)[1])[0], 'r-', lw=2, label = 'Optimized fit')
	plt.plot(peak_freqs, peak_amplitudes, 'ro', markersize=10, label = 'Peaks')


	fitting = True
	if fitting and len(peak_indices) > 0:

		print 'Number of peaks = ', len(peak_indices)

		print 'Output of fit_peaks:', fit_peaks(data), '########'

		if len(peak_indices)==1:
			output = fit_peaks(data)
			(amplitude,centre,sigma,slope,c) = output[0]
			(amplitude_err, centre_err, sigma_err, slope_err, c_err) = output[1]
			peak_plotting_freqs = np.arange(centre-3*sigma,centre+3*sigma,1)
			fit = gaussian(peak_plotting_freqs,amplitude,centre,sigma) + line(peak_plotting_freqs, slope, c)
			plt.plot(peak_plotting_freqs, fit, label='Gaussian', color='black')

		else:
			for i in range(len(peak_indices)):
				output = fit_peaks(data)
				(amplitude,centre,sigma,slope,c) = output[0][i,:] 
				(amplitude_err, centre_err, sigma_err, slope_err, c_err) = output[1][i,:] 
				percent_err = 100 * output[1][i,:]/output[0][i,:]
				print 'percentage errors = ', percent_err
				#amplitude = amplitude*np.sqrt(2*np.pi)*sigma
				peak_plotting_freqs = np.arange(centre-3*sigma,centre+3*sigma,1)
				fit = gaussian(peak_plotting_freqs,amplitude,centre,sigma) + line(peak_plotting_freqs, slope, c)
				plt.plot(peak_plotting_freqs, fit, label='Gaussian'+str(i+1), color='black')
	plt.xlabel(r'$\omega$ / $Hz$', fontsize = 18)    
	plt.ylabel('Intensity $(a.u.)$', fontsize = 18)
	plt.legend()
plt.show()