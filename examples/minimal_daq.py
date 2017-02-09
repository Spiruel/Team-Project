# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 10:46:02 2016

@author: lcfl72

Example script for thermal properties experiments

Controls:
    NI DAQ analog input and output - e.g. NI USB-6009

"""
from __future__ import print_function, division
from daqai import DAQmx_ReadAI as ReadAI

import numpy
import pylab
import sys

from time import sleep

# Get required measurement duration and output filename from commandline
#DURATION = 1.0
DURATION = float(sys.argv[1])
#savename = None
savename = sys.argv[2]

# Hardcoded values: these may need to be changed or obtained from commandline
SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

AICHANNELS = "Dev1/ai0:3"	# Analog input channels
NCHANS = 3			# number of analog input channels
VMax = 10.0					# Maximum input voltage 

n_reps = 1
# Initialise data arrays
aves = numpy.zeros((n_reps,4))
errs = numpy.zeros((n_reps,4))

   
	
# Read in data from analog input channels into new numpy array 'data'
data = ReadAI(DURATION, chanlist=AICHANNELS, nchans=NCHANS, samplerate=SAMPLE_RATE, vrange=VMax)

# Manipulate the data to 'calibrate' the varous quantities you are measuring
data[:,0] = 0 + 1*data[:,0] # dataset for channel ai0
data[:,1] = 0 + 1*data[:,1] # dataset for channel ai1
data[:,2] = 0 + 1*data[:,2] # dataset for channel ai2
#data[:,3] = 0 + 1*data[:,3] # dataset for channel ai3

pylab.plot(data[:,0])
pylab.show()
# Once calibrations are done, can find mean for each quantity
# aves[pt] = numpy.mean(data, axis=0, dtype=numpy.float64)

# Print values to terminal
# print('datapoint {} of {}.'.format(pt+1, n_reps))
# for i in range(4):
	# print('Dev1/ai{} : val = {} V'.format(i, aves[pt][i]))

# Save to file:
# if not (savename == None): 
	# with open(savename, 'w') as f:					# Write to file as comma separated - you may well want 
													# to modify this to 'append' data ('a') rather than 
													# overwriting ('w').
		# print(', '.join(map(str,aves[pt]))+', '+', '.join(map(str,errs[pt])), file=f)

# Plot datasets        
# pylab.plot(aves, marker='x')
# pylab.show()