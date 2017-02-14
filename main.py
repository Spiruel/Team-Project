import time
import threading
from Tkinter import *
import sys
import numpy as np
import matplotlib.pyplot as plt

from daqai import DAQmx_ReadAI as ReadAI

class DataCaptThread(threading.Thread):
	def __init__(self, simulated=False):
		threading.Thread.__init__(self)

		self.DURATION = .05
		self.savename = None

		# Hardcoded values: these may need to be changed or obtained from commandline
		self.SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

		self.AICHANNELS = "Dev1/ai0:2"	# Analog input channels
		self.NCHANS = 3			# number of analog input channels
		self.VMax = 10.0			# Maximum input voltage
		
		self.d1 = np.array([]); self.d2 = np.array([]); self.d3 = np.array([]) #should be queue
		
		self.running = True
		self.simulated = simulated
		
		self.block_size = int(self.DURATION*self.SAMPLE_RATE)
		
	def run(self):
		if not self.simulated:
			try:
				while self.running:
					data = ReadAI(self.DURATION, chanlist=self.AICHANNELS, nchans=self.NCHANS, samplerate=self.SAMPLE_RATE, vrange=self.VMax)

					self.d1 = np.concatenate((self.d1, data[:,0]))
					self.d2 = np.concatenate((self.d2, data[:,1]))
					self.d3 = np.concatenate((self.d3, data[:,2]))

			except KeyboardInterrupt:
				sys.exit()
		else:
			full_data = np.loadtxt('data/mondaylongruntest.csv', delimiter=',', comments='#') #simulates the measurement of data from a csv defined here
			try:
				i = 0
				while self.running:
					
					data = full_data[i:i+self.block_size]
					i += self.block_size
					time.sleep(self.DURATION)
					self.d1 = np.concatenate((self.d1, data[:,0]))
					self.d2 = np.concatenate((self.d2, data[:,1]))
					self.d3 = np.concatenate((self.d3, data[:,2]))

			except KeyboardInterrupt:
				sys.exit()
	def stop(self):
		self.running = False
		
class Analysis():
	def __init__(self, test):
		print 'Started analysis'
		self.data = data_capture
		plt.ion()
		plt.ylim([-0.6,0.6])
		plt.ylabel('Amplitude / V')
		plt.xlabel('Samples')
		
	def show_data(self):
		plt.plot(self.data.d1, 'r-'); plt.plot(self.data.d2, 'g-', label='y'); plt.plot(self.data.d3, 'b-', label='z')
		
		win_size = self.data.SAMPLE_RATE*self.data.DURATION*10
		plt.xlim([0, win_size])
		if len(self.data.d1) >= win_size:
			plt.xlim([len(self.data.d1)-win_size, len(self.data.d1)])
		plt.pause(0.05)
		
		block_len = self.data.DURATION*self.data.SAMPLE_RATE
		print 'Channel 1 std :', np.std(self.data.d1[-block_len:])
		

if __name__ == "__main__":

	time.sleep(.5)
	
	if len(sys.argv) > 1:
		if sys.argv[1].lower() == 'simulated':
			data_capture = DataCaptThread(simulated=True)
		else:
			data_capture = DataCaptThread()
	else:
		data_capture = DataCaptThread()
	
	data_capture.start()
	analysis = Analysis(data_capture)
	
	try:
		while True:
			analysis.show_data()
	except KeyboardInterrupt:
		print 'Saving data and exiting!'
		data_capture.stop()
		filename = raw_input('Enter filename here: ')
		np.savetxt('data/'+filename+'.csv', np.c_[data_capture.d1,data_capture.d2,data_capture.d3], delimiter=',')
		print 'Finished saving'
