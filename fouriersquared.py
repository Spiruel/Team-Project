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
n = len(time)
k = np.arange(n)
T = n / sample_rate
frequency = k / T
frequency = frequency[range(n/2)]

    
Y = fft(amplitude) / n
Y = abs(Y[range(n/2)])
#    Y = 20*np.log10(Y)
plt.plot(frequency, Y)    
    
def lorentzian(p):
    num = (p[0] ** 2)
    denom = ((frequency - (p[1])) ** 2 + p[0] ** 2)
    Y = p[2] * (num / denom)
    return Y
    
def residuals(p, amplitude):
    err = Y - lorentzian(p)
    return err

    #background subtraction

ind_bg_low = (frequency > min(frequency)) & (frequency < 299.0) #defining background
ind_bg_high = (frequency > 301.25) & (frequency < max(frequency))

frequency_bg = np.concatenate((frequency[ind_bg_low], frequency[ind_bg_high]))
Y_bg = np.concatenate((Y[ind_bg_low], Y[ind_bg_high]))
        #plt.plot(x_bg, y_bg)

    #fitting backround to a line
m, c = np.polyfit(frequency_bg, Y_bg, 1)

    #removing fitted background
background = m * frequency + c
Y_bg_corr = Y - background
plt.plot(frequency, background)

    #initial values
p = [0.030, 300.1, 0.0007848] #hwhm, peak centre, intensity
#p = [0.01, 63.0354, 0.0046] #x axis 10 mins data

    #optimization
pbest = leastsq(residuals, p, args = (Y_bg_corr), full_output = 1)
best_parameters = pbest[0]


    #fit to data
fit = lorentzian(best_parameters)

#plt.plot(frequency, amplitude_bg_corr, 'wo')
plt.plot(frequency, lorentzian(best_parameters) + background, 'r-', lw=2)
plt.xlabel(r'$\omega$ (cm$^{-1}$', fontsize = 18)
plt.ylabel('Intensity (a.u.)', fontsize = 18)
#plt.xlim(298.5, 302)
#plt.ylim(0.0, 0.0009)


plt.show()