import time
import threading
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import argparse

try:
	from daqai import DAQmx_ReadAI as ReadAI
except:
	print 'Cannot import daqai. Try using simulated mode instead?'
	
import algorithms

class DataCaptThread(threading.Thread):
	def __init__(self, filename='data/testinwater.csv', simulated=False):
		threading.Thread.__init__(self)

		self.DURATION = .05
		self.savename = None

		# Hardcoded values: these may need to be changed or obtained from commandline
		self.SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

		self.AICHANNELS = "Dev1/ai0:2"	# Analog input channels
		self.NCHANS = 3			# number of analog input channels
		self.VMax = 10.0			# Maximum input voltage
		
		self.data = np.array([])
		self.d1 = np.array([]); self.d2 = np.array([]); self.d3 = np.array([]) #should be queue
		
		self.running = True
		self.filename = filename
		self.simulated = simulated
		
		self.block_size = int(self.DURATION*self.SAMPLE_RATE)
		
	def run(self):
		if not self.simulated:
			try:
				while self.running:
					try:
						self.data = ReadAI(self.DURATION, chanlist=self.AICHANNELS, nchans=self.NCHANS, samplerate=self.SAMPLE_RATE, vrange=self.VMax)
					except:
						print 'Something went wrong. DAQ did not capture data. Try simulated mode?'
						self.stop()
						sys.exit(1)
						
					self.d1 = np.concatenate((self.d1, self.data[:,0]))
					self.d2 = np.concatenate((self.d2, self.data[:,1]))
					self.d3 = np.concatenate((self.d3, self.data[:,2]))

			except KeyboardInterrupt:
				sys.exit()
		else:
			time_start = time.time()
			full_data = np.loadtxt(self.filename, delimiter=',', comments='#') #simulates the measurement of data from a csv defined here
			total_size = len(full_data)
			try:
				i = 0
				while self.running:
					
					self.data = full_data[i:i+self.block_size]
					i += self.block_size
					time.sleep(self.DURATION)
					self.d1 = np.concatenate((self.d1, self.data[:,0]))
					self.d2 = np.concatenate((self.d2, self.data[:,1]))
					self.d3 = np.concatenate((self.d3, self.data[:,2]))
					
					if i+self.block_size > total_size:
						print 'Time taken:', time.time()-time_start, 's. Time actual:', str(len(full_data[:,0])/self.SAMPLE_RATE), 's.'
						self.stop()

			except KeyboardInterrupt:
				sys.exit()
	def stop(self):
		print 'Stopping data capture!'
		self.running = False
		
class Analysis():
	def __init__(self, data_capture, plot):
		print 'Started analysis'
		self.data = data_capture
		
		self.show_plots = plot
		
		if self.show_plots:	
			self.win_size = self.data.SAMPLE_RATE*self.data.DURATION*20 #define window size of plot area
			
			self.active_plots = [0]
			
			self.fig = plt.figure()

			# Draw the plot
			self.ax = plt.subplot2grid((6, 1), (0, 0), rowspan=5)
			self.ax1 = plt.subplot2grid((6, 1), (5, 0))

			self.fig.subplots_adjust(left=0.25)
			
			plt.ion()
			self.ax.set_ylim([-0.6,0.6])
			self.ax.set_ylabel('Amplitude / V')
			self.ax1.set_ylabel('Anomaly \nScore')
			self.ax1.set_xlabel('Samples')
			self.ax.get_xaxis().set_ticks([])
			self.ax.set_xlim([0, self.win_size])
			self.ax1.set_xlim([0, self.win_size])
		
			[self.line] = self.ax.plot([0], [0], 'r-')
			[self.line1] = self.ax.plot([0], [0], 'g-')
			[self.line2] = self.ax.plot([0], [0], 'b-')
			
			# Add a set of radio buttons for changing colour
			self.color_radios_ax = self.fig.add_axes([0.025, 0.5, 0.1, 0.35], axisbg='lightgoldenrodyellow')
			self.color_radios = RadioButtons(self.color_radios_ax, ('0', '1', '2', '0,1,2'), active=0) #need to add '0,1', '0,2', '1,2', '0,1,2'
			self.color_radios.on_clicked(self.color_radios_on_clicked)
			
	def color_radios_on_clicked(self, choice):
		colours = {0:'r', 1:'g', 2:'b'}
		
		choice = [int(i[-1]) for i in choice.split(',')]
		if choice != self.active_plots:
			self.active_plots = choice
			
			if len(choice) == 1:
				self.line.set_color(colours[choice[0]])
			else:
				self.line.set_color('r')
				self.line1.set_color('g')
				self.line2.set_color('b')
				
		self.fig.canvas.draw_idle()
		
	def show_data(self):
		if self.data.running:
		
			if self.data.simulated:
				while np.isnan(np.std(self.data.d1)):
					print 'Waiting to load data...'
					while np.isnan(np.std(self.data.d1)):
						pass
		
			med_abs_dev_mask = algorithms.median_absolute_deviation(self.data.data) #self.data.data is the captured data in last duration
			grubbs_mask = algorithms.grubbs(self.data.data)
			five_sig_mask = algorithms.five_sigma(self.data.data)
			dev_mov_av_mask = algorithms.stddev_from_moving_average(self.data.d1[-self.data.block_size:])			
			
			anom_tests = [med_abs_dev_mask, five_sig_mask, grubbs_mask] #dev_mov_av_mask

			anom_ys = np.array([np.zeros(self.data.block_size), np.zeros(self.data.block_size), np.zeros(self.data.block_size)])
			
			d1_anom_ys = np.array([])
			d2_anom_ys = np.array([])
			d3_anom_ys = np.array([])
			
			for channel in range(3):
				for test in anom_tests:
					if len(test) == 0:
						anom_occurrences = np.array([0])
					else:
						anom_occurrences = np.bincount(test[channel])
					if len(anom_occurrences) < self.data.block_size:
						anom_occurrences = np.append(anom_occurrences, np.zeros(self.data.block_size - len(anom_occurrences)))
					anom_ys[channel] = np.add(anom_ys[channel], anom_occurrences)	

			#mask = med_abs_dev_mask[0]

			'''sum_anom = 0
			for test in anom_tests: #med_abs_dev_mask
				sum_anom += len(i)
			
			if sum_anom > 0: 
				print 'Anomaly alert!'''
				
			if self.show_plots:
			
				active_plots = set(self.active_plots)
				if active_plots == set([0]):
					ys = self.data.d1
				if active_plots == set([1]):
					ys = self.data.d2
				if active_plots == set([2]):
					ys = self.data.d3
				if set([0,1]) == active_plots: #multi choice doesnt work yet
					ys = self.data[:,[0, 1]]
				if set([0,2]) == active_plots:
					ys = self.data[:,[0, 2]]
				if set([1,2]) == active_plots:
					ys = self.data[:,[1, 2]]
				
				size = int(self.win_size)
				if set([0,1,2]) != active_plots:
					if len(ys) <= size:
						xs = np.linspace(0, len(ys), len(ys))
					else:
						total_dur = len(ys)
						xs = np.linspace(total_dur-size, total_dur, size)
						ys = ys[-size:] #need to make ys not crop from RHS but rather window across from left. at risk of overplotting as capture thread sleeps but show does not
				else:
				
					for ys, l in zip([self.data.d2, self.data.d3, self.data.d1], [self.line1, self.line2, self.line]):
						if len(ys) <= size:
							xs = np.linspace(0, len(ys), len(ys))
						else:
							total_dur = len(ys)
							xs = np.linspace(total_dur-size, total_dur, size)
							ys = ys[-size:]
						if l != self.line:
							l.set_xdata(xs)
							l.set_ydata(ys)
				
				self.line.set_xdata(xs)
				self.line.set_ydata(ys)
					
				self.ax1.plot(xs[-self.data.block_size:], anom_ys[0], 'r-')
				self.ax1.plot(xs[-self.data.block_size:], anom_ys[1], 'g-')
				self.ax1.plot(xs[-self.data.block_size:], anom_ys[2], 'b-')
				# import Filters as F
				# if not np.isnan(np.std(self.data.d1)):
					# self.ax.plot(F.movingaverage(self.data.d1, 0.01*data_capture.block_size), 'k--')

				for ax in [self.ax, self.ax1]:
					if len(self.data.d1) >= self.win_size:
						ax.set_xlim([len(self.data.d1)-self.win_size, len(self.data.d1)])
				plt.pause(self.data.DURATION)

if __name__ == "__main__":
	time.sleep(.5)
	
	parser = argparse.ArgumentParser(description='Control simulation or plotting optoions')
	parser.add_argument('-p', '--plot', action='store_true', help='enables plot view')
	parser.add_argument('-s', '--simulated', type=str, nargs='*', help='file for simulated input', required=False, default='default')

	args = parser.parse_args()
	
	input = args.simulated
	capturing = False
	if input == 'default':
		data_capture = DataCaptThread()
	else:
		if len(input) == 1:
			input = input[0]
			if input[-4:] != '.csv': 
				input += '.csv'
			if os.path.isfile('data/'+input):
				filename = input
				print 'Found file,', str(input) + ', choosing it for input...'
				data_capture = DataCaptThread(filename='data/'+filename, simulated=True)
				capturing = True
		if not capturing:
			print 'Couldn\'t find file,', str(input) + ', choosing default input file...'
			data_capture = DataCaptThread(simulated=True)
	
	data_capture.start()
	analysis = Analysis(data_capture, args.plot)
	
	try:
		while data_capture.running:
			analysis.show_data()
	except KeyboardInterrupt:
		print 'Saving data and exiting!'
		data_capture.stop()
		filename = raw_input('Enter filename here: ')
		np.savetxt('data/'+filename+'.csv', np.c_[data_capture.d1,data_capture.d2,data_capture.d3], delimiter=',')
		print 'Finished saving'
