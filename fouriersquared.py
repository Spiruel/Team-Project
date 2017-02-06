import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft

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
    Y = Y[range(n/2)]
    
    plt.plot(frequency, abs(Y))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    
plotFourierSquared(amplitude, sample_rate)
plt.show()