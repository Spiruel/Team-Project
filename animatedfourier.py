import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def waveform(data, fs = 1500):
	'''data is 1d array of amplitudes for a given time. fs is sampling frequency of the signal'''
	data_len = len(data)
	time_array = np.arange(data_len)/fs
	return (time_array,data)
	
data = np.loadtxt(r'D:\Users\Samuel\Desktop\TeamProject\data\testinwater.csv', delimiter = ',', usecols=[0])

SAMPLE_RATE = 3000
DURATION = 500

size = SAMPLE_RATE * DURATION

# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# fig, ax = plt.subplots()
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.show()

[line] = plt.plot([0],[0], 'k-')
x0 = 0

plt.ion()
for i in range(len(data)):
	x, y = waveform(data[i:i+size]) #get time and amplitude array of data here
	#fourier transform here and plot it
	x0 += x[-1]
	x += x0
	plt.plot(x,y)
	# plt.xlim([x0,x0+size])
	line.set_xdata(x)
	line.set_ydata(y)
	plt.pause(0.05)