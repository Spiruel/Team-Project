import time
import threading
from Tkinter import *
import sys
import numpy as np
import matplotlib.pyplot as plt

from daqai import DAQmx_ReadAI as ReadAI

class DataCaptThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		
		self.DURATION = .05
		self.savename = None

		# Hardcoded values: these may need to be changed or obtained from commandline
		self.SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

		self.AICHANNELS = "Dev1/ai0:2"	# Analog input channels
		self.NCHANS = 3			# number of analog input channels
		self.VMax = 10.0			# Maximum input voltage
		
		self.d1 = np.array([]); self.d2 = np.array([]); self.d3 = np.array([]) #should be queue
		
	def run(self):
		try:
			while True:
				data = ReadAI(self.DURATION, chanlist=self.AICHANNELS, nchans=self.NCHANS, samplerate=self.SAMPLE_RATE, vrange=self.VMax)

				self.d1 = np.concatenate((self.d1, data[:,0]))
				self.d2 = np.concatenate((self.d2, data[:,1]))
				self.d3 = np.concatenate((self.d3, data[:,2]))
		except KeyboardInterrupt:
			sys.exit()
		
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
		

if __name__ == "__main__":

	time.sleep(.5)
	data_capture = DataCaptThread()
	data_capture.start()
	analysis = Analysis(data_capture)
	
	try:
		while True:
			analysis.show_data()
	except KeyboardInterrupt:
		print 'Saving data and exiting!'
		filename = raw_input('Enter filename here: ')
		np.savetxt('data/'+filename+'.csv', np.c_[data_capture.d1,data_capture.d2,data_capture.d3], delimiter=',')
		print 'Finished saving'

	
	# try:
		# while True:
			# print len(data_capture.d1)
	# except KeyboardInterrupt:
		# data_capture.quit()
		# print 'hello'
		# data_capture.stop()
