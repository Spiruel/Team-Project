import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.loadtxt('Team 17/sin wave test.csv', delimiter=',')
time, amplitude = data[:,0], data[:,1]

plt.plot(time, amplitude)
plt.show()

ps = np.abs(np.fft.fft(amplitude))**2

f, Pxx_den = signal.periodogram(amplitude, 1000)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()