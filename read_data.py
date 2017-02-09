from __future__ import division
from daqai import DAQmx_ReadAI as ReadAI

import numpy as np
import matplotlib.pyplot as plt
import sys

from time import sleep
import sys

# Get required measurement duration and output filename from commandline
#DURATION = 1.0
DURATION = .05
savename = None
#savename = sys.argv[2]

# Hardcoded values: these may need to be changed or obtained from commandline
SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

AICHANNELS = "Dev1/ai0:2"	# Analog input channels
NCHANS = 3			# number of analog input channels
VMax = 10.0					# Maximum input voltage 

plt.ion()

fig=plt.figure()
# line = plt.plot(np.linspace(0,SAMPLE_RATE*DURATION,SAMPLE_RATE*DURATION), np.linspace(0,SAMPLE_RATE*DURATION,SAMPLE_RATE*DURATION),'r-')
# line1 = plt.plot(np.linspace(0,DURATION,SAMPLE_RATE*DURATION),np.linspace(0,DURATION,SAMPLE_RATE*DURATION),'b-')
# line2 = plt.plot(np.linspace(0,DURATION,SAMPLE_RATE*DURATION),np.linspace(0,DURATION,SAMPLE_RATE*DURATION),'g-')
plt.ylim([-0.6,0.6])
plt.ylabel('Amplitude / V')
plt.xlabel('Samples')

d1 = np.array([]); d2 = np.array([]); d3 = np.array([])
while True:
	try:
		data = ReadAI(DURATION, chanlist=AICHANNELS, nchans=NCHANS, samplerate=SAMPLE_RATE, vrange=VMax) #NEED TO THREAD THIS

		d1 = np.concatenate((d1, data[:,0])); d2 = np.concatenate((d2, data[:,1])); d3 = np.concatenate((d3, data[:,2]))
		plt.plot(d1, 'r-'); plt.plot(d2, 'b-'); plt.plot(d3, 'g-') #make this bit interactive
		
		# line[0].set_xdata(np.linspace(0,SAMPLE_RATE*DURATION,SAMPLE_RATE*DURATION))
		# line1[0].set_ydata(d2)
		# line2[0].set_ydata(d3)
		
		win_size = SAMPLE_RATE*DURATION*10
		plt.xlim([0, win_size])
		if len(d1) >= win_size:
			plt.xlim([len(d1)-win_size, len(d1)])
		plt.pause(0.05)
	
	except KeyboardInterrupt:
		print 'Saving data and exiting!'
		filename = raw_input('Enter filename here: ')
		np.savetxt('data/'+filename+'.csv', np.c_[d1,d2,d3])
		sys.exit(1)