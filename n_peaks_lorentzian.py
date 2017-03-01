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

def line(x, slope, intercept):
    "line"
    return slope * x + intercept


def lorentz_params(data):
	'''
	Function to return the half-width half-maximum, peak centre and peak intensity
	of the Lorentzian given a data set
	'''
	amplitude = data #[:,0] #Need to change this so that it works for each column of data, not just the first
	
	sample_rate = 3000.0
	n = len(amplitude)
	k = np.arange(n)
	T = n / sample_rate
	frequency = k / T
	frequency = frequency[range(np.int(n/2))] #cutting the array off at half way becuase of the Nyquist limit
	lowcut = 10
	Y = fourier_transform(amplitude, sample_rate, lowcut)
	Y_av = Filters.movingaverage(Y, 50)    
	p = [10, peak_finder(frequency, Y_av, sample_rate)[0], peak_finder(frequency, Y_av, sample_rate)[1]] #hwhm, peak centre, intensity    
		
	pbest = leastsq(residuals, p, args = (Y_av, frequency), full_output = 1)
	best_parameters = pbest[0]
	maxindex_Y, maxvalue_Y = max(enumerate(Y_av), key=operator.itemgetter(1))
	p_new = (best_parameters[0], best_parameters[1], maxvalue_Y)
	return p_new
	
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
Function returns the Lorentzian fit on the parameters supplied
'''
def lorentzian(frequencies, amplitude, centre, sigma):
	lorentzian = amplitude/np.pi * sigma/((frequencies-centre)**2 + simga**2)
	return lorentzian
	
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
	 
if __name__ == '__main__':
	data = np.loadtxt('data/12v_comparisontobaseline.csv', delimiter=',', comments='#')[:,0][5000:10000]
	frequencies, sample_rate, amplitudes = params(data)
	
	peak_indices, peak_freqs, peak_amplitudes = find_peaks(data)
	peak_freq1 = peak_freqs[0]
	peak_amp1 = peak_amplitudes[0]

	peak_index1 = conv_to_samples(data,peak_freqs[0])


	#####finding constant offset

	mod = Model(gaussian) + Model(line)
	pars  = mod.make_params( amp=peak_amp1, cen=peak_freq1, wid=20, slope=0, intercept=0)

	out = mod.fit(amplitudes[conv_to_samples(data,peak_freq1-50):conv_to_samples(data,peak_freq1+50)], pars, x=frequencies[conv_to_samples(data,peak_freq1-50):conv_to_samples(data,peak_freq1+50)])

	'''mod = GaussianModel()

	pars = mod.guess(amplitudes, x=frequencies)
	out  = mod.fit(amplitudes, pars, x=frequencies)'''

	amplitude = out.params['amp'].value
	centre = out.params['cen'].value
	sigma = out.params['wid'].value
	slope = out.params['slope'].value
	c = out.params['intercept'].value

	print out.fit_report(min_correl=0.25)

	print (amplitude, centre, sigma, slope, c)

	fit = gaussian(frequencies,amplitude,centre,sigma) + line(frequencies, slope, c)

	#    hwhm, peak_cen, peak_inten = lorentz_params(data)
	#    xs = np.linspace(0,700,100)
	#    plt.plot(xs, lorentzian(xs, (hwhm, peak_cen, peak_inten)))
	plt.plot(params(data)[0], fourier_transform(data, 3000, 100), label = 'Raw data')
	plt.plot(params(data)[0], amplitudes, label = 'Smoothed data')
	plt.plot(frequencies, fit, label='Gaussian', color='red')
	#plt.plot(params(data)[0], optimization(params(data)[0], lorentz_params(data), params(data)[2], params(data)[1]) + background_subtraction(params(data)[2], params(data)[0], params(data)[1])[0], 'r-', lw=2, label = 'Optimized fit')
	plt.plot(peak_freqs, peak_amplitudes, 'ro', markersize=10, label = 'Peaks')
	plt.xlabel(r'$\omega$ / $Hz$', fontsize = 18)    
	plt.ylabel('Intensity $(a.u.)$', fontsize = 18)
	plt.legend()
plt.show()