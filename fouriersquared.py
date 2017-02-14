import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.optimize import leastsq
import operator
#import Filters
from scipy.signal import butter, lfilter


data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')
time, amplitude = data[:,0], data[:,1]

sample_rate = 1500.0
sample_interval = 1.0/1500.0
n = len(time)
k = np.arange(n)
T = n / sample_rate
frequency = k / T
frequency = frequency[range(n/2)]
lowcut = 100
highcut = 350


def butter_highpass(lowcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    b, a = butter(order, low, btype ='high')
    return b, a
    
    
def butter_highpass_filter(data, lowcut, sample_rate, order=5):
    b, a = butter_highpass(lowcut, sample_rate, order)
    y = lfilter(b, a, data)
    return y
    

def fourier_transform(amplitude, sample_rate):
    dat = butter_highpass_filter(amplitude, lowcut, sample_rate)
    Y = fft(dat) / n
    Y = abs(Y[range(n/2)])
    return Y

def peak_finder(sample_rate):
    maxindex_Y, maxvalue_Y = max(enumerate(Y_av), key=operator.itemgetter(1))
    return frequency[maxindex_Y], maxvalue_Y


def lorentzian(frequency, p):
    num = (p[0] ** 2)
    denom = ((frequency - (p[1])) ** 2 + p[0] ** 2)
    YL = p[2] * (num / denom)
    return YL
    
    
def residuals(p, Y, frequency):
    err = Y_av - lorentzian(frequency, p)
    return err
    

def background_subtraction(frequency):
    ind_bg_low = (frequency > min(frequency)) & (frequency < 299.0) #defining background
    ind_bg_high = (frequency > 301.25) & (frequency < max(frequency))
    frequency_bg = np.concatenate((frequency[ind_bg_low], frequency[ind_bg_high]))
    Y_bg = np.concatenate((Y_av[ind_bg_low], Y_av[ind_bg_high]))
    m, c = np.polyfit(frequency_bg, Y_bg, 1)
    background = m * frequency + c
    Y_bg_corr = Y_av - background
    return background, Y_bg_corr


def optimization(frequency, p):
    pbest = leastsq(residuals, p, args = (background_subtraction(frequency)[1], frequency), full_output = 1)
    best_parameters = pbest[0]
    p_new = (best_parameters[0], best_parameters[1], peak_finder(sample_rate)[1] - background_subtraction(frequency)[0])
    fit = lorentzian(frequency, p_new)
    return fit
    
    
def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')
	
if __name__ == '__main__':
    
    Y = fourier_transform(amplitude, sample_rate)
    

    Y_av = movingaverage(Y, 15)
    
    p = [0.010, peak_finder(sample_rate)[0], peak_finder(sample_rate)[1]] #hwhm, peak centre, intensity
    
    plt.plot(frequency, Y, label = 'Data')
    plt.plot(frequency, Y_av, label = 'Smoothed data')
    plt.plot(frequency, background_subtraction(frequency)[0], label = 'Background')
    plt.plot(frequency, optimization(frequency, p) + background_subtraction(frequency)[0], 'r-', lw=2, label = 'Optimized fit')
#    plt.plot(frequency, lorentzian(frequency, p) + background_subtraction(frequency)[0], c = 'black', label = 'First fit')
    plt.scatter(peak_finder(sample_rate)[0], peak_finder(sample_rate)[1], c = 'r', alpha = 0.5)
    plt.xlabel(r'$\omega$ $(cm^{-1})$', fontsize = 18)
    plt.ylabel('Intensity $(a.u.)$', fontsize = 18)
#    plt.xlim(55, 70)
#    plt.ylim(0.0, 0.001)
    plt.legend()
    print optimization(frequency, p), lorentzian(frequency, p)

plt.show()
