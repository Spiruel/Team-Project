import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft

from STFT import STFT

data = np.loadtxt('Team 17/12V motor x axis ten minutes.csv', delimiter=',', comments='#')
time, amplitude = data[:,0], data[:,1]

freq = 1500
tim_per = 2

step = tim_per*freq

step = tim_per*freq
'''
t, a = np.array([0,1]), np.array([0,1])
line, = plt.plot(t, a, 'k')
for i in range(0,len(amplitude),1):
	t, a = time[0:i+step], amplitude[0:i+step]
	#plt.plot(t[-2:], a[-2:], 'k')

	line.set_xdata(t); line.set_ydata(a)
	plt.draw()
	plt.pause(0.05)


	scroll_size = .1
	xlim = t[-1]
	if xlim<=scroll_size:
		plt.xlim([0, scroll_size])
	else:
		plt.xlim([xlim-scroll_size, xlim])'''

def animate_waterfall(amplitude, step):

	plt.ion()
	
	#result = np.ndarray(shape=(12,470))
	plt.xlabel('time'); plt.ylabel('freq')
	for i in range(0,len(amplitude),step):
		#result = np.concatenate((result, STFT(amplitude[i:i+step], fs=freq, clip=True)[0]), axis=0)
		result = STFT(amplitude[0:i+step], fs=freq, clip=True)[0]
		img = plt.imshow(result.T, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')

		if i == 0: plt.colorbar()
		plt.pause(0.05)
		#plt.savefig('test/'+str(i)+'.png')

		scroll_size = 200
		xlim = result.shape[0]
		if xlim<=scroll_size:
			plt.xlim([0, scroll_size])
		else:
			plt.xlim([xlim-scroll_size, xlim])

step = tim_per*freq
animate_waterfall(amplitude, step)
# plt.plot(time, amplitude)
# plt.show()

# for i in range(0,len(amplitude),step):
	# print stats.skew(amplitude[0:i+step]), stats.kurtosis(amplitude[0:i+step], fisher=False)
	
'''
plt.plot(time, amplitude)
plt.show()

ps = np.abs(np.fft.fft(amplitude))**2

f, Pxx_den = signal.periodogram(amplitude, 1000)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
'''
