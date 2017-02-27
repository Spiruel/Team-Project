import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.optimize import leastsq
import Filters
import operator
from scipy.signal import butter, lfilter

'''
Function to return the half-width half-maximum, peak centre and peak intensity
of the Lorentzian given a data set
'''
def lorentz_params(data):
    amplitude = data[:,0]
    
    sample_rate = 3000.0
    n = len(amplitude)
    k = np.arange(n)
    T = n / sample_rate
    frequency = k / T
    frequency = frequency[range(n/2)]
    lowcut = 50
    Y = fourier_transform(amplitude, sample_rate, lowcut)
    Y_av = Filters.movingaverage(Y, 15)    
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
    amplitude = data[:,0]
    
    sample_rate = 3000.0
    n = len(amplitude)
    k = np.arange(n)
    T = n / sample_rate
    frequency = k / T
    frequency = frequency[range(n/2)]
    lowcut = 100
    Y = fourier_transform(amplitude, sample_rate, lowcut)
    Y_av = Filters.movingaverage(Y, 10) 
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
    Y = abs(Y[range(len(amplitude)/2)])
    return Y

'''
Function returning the location and intensity of the peak in the data
'''
def peak_finder(frequency, Y_av, sample_rate):
    maxindex_Y, maxvalue_Y = max(enumerate(Y_av), key=operator.itemgetter(1))
    return frequency[maxindex_Y], maxvalue_Y, maxindex_Y

'''
Function returns the Lorentzian fit on the parameters supplied
'''
def lorentzian(frequency, p):
    num = (p[0] ** 2)
    denom = ((frequency - (p[1])) ** 2 + p[0] ** 2)
    YL = p[2] * (num / denom)
    return YL
    
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
	data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')
	print lorentz_params(data)
#    hwhm, peak_cen, peak_inten = lorentz_params(data)
#    xs = np.linspace(0,700,100)
#    plt.plot(xs, lorentzian(xs, (hwhm, peak_cen, peak_inten)))
	plt.plot(params(data)[0], fourier_transform(data[:,0], 3000, 100), label = 'Raw data')
	plt.plot(params(data)[0], params(data)[2], label = 'Smoothed data')
	plt.plot(params(data)[0], optimization(params(data)[0], lorentz_params(data), params(data)[2], params(data)[1]) + background_subtraction(params(data)[2], params(data)[0], params(data)[1])[0], 'r-', lw=2, label = 'Optimized fit')
	plt.xlabel(r'$\omega$ $(cm^{-1})$', fontsize = 18)    
	plt.ylabel('Intensity $(a.u.)$', fontsize = 18)
	plt.legend()
plt.show()