import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.optimize import leastsq

from STFT import STFT

data = np.loadtxt('Team 17/sin wave test.csv', delimiter=',', comments='#')
time, amplitude = data[:,0], data[:,1]

sample_rate = 1500.0
sample_interval = 1.0/1500.0

#def waveform(data, sample_rate):
#    data_len = len(data)
#    time_values = data_len / sample_rate
#    waveform = np.array(data, time_values)
#    return waveform

def plotFourierSquared(amplitude, sample_rate):
    n = len(time) #length of signal
    k = np.arange(n)
    T = n / sample_rate
    frequency = k / T
    frequency = frequency[range(n/2)]
    
    Y = fft(amplitude) / n
    Y = abs(Y[range(n/2)])
#    Y = 20*np.log10(Y)
    
    plt.plot(frequency, Y)
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    
def lorentzian(p):
    num = (p[0]**2)
    denom = (frequency - (p[1])) ** 2 + p[0] ** 2
    amplitude = p[2] * (num / denom)
    return amplitude
    
def residuals(p, amplitude):
    err = amplitude - lorentzian(p)
    return err

    #background subtraction

ind_bg_low = (time > min(time)) & (time < 299.0) #defining background
ind_bg_high = (time > 301.25) & (time < max(time))

time_bg = np.concatenate((time[ind_bg_low], time[ind_bg_high]))
amplitude_bg = np.concatenate((amplitude[ind_bg_low], amplitude[ind_bg_high]))
        #plt.plot(x_bg, y_bg)

    #fitting backround to a line
m, c = np.polyfit(time_bg, amplitude_bg, 1)

    #removing fitted background
background = m * time + c
amplitude_bg_corr = amplitude - background
        #plt.plot(x, y_bg_corr)

    #initial values
p = [0.020, 300.107, 0.0007248] #hwhm, peak centre, intensity

    #optimization
pbest = leastsq(residuals, p, args = (amplitude_bg_corr), full_output = 1)
best_parameters = pbest[0]

    #fit to data
fit = lorentzian(best_parameters)

#plt.plot(time, amplitude_bg_corr, 'wo')
plt.plot(time, lorentzian(p), 'r-', lw=2)
plt.xlabel(r'$\omega$ (cm$^{-1}$', fontsize = 18)
plt.ylabel('Intensity (a.u.)', fontsize = 18)

print time

plotFourierSquared(amplitude, sample_rate)
plt.show()