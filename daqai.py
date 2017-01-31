from __future__ import division, print_function

from PyDAQmx import *

import numpy as np
import pylab
import sys


def DAQmx_ReadAI(meas_duration, chanlist='Dev1/ai0', nchans=1, samplerate=10000, vrange=1.0):
    '''
    Read analog input(s) from NIDAQmx device(s). 
        meas_duration is the measurement duration (float, in seconds) for acquisition
        chanlist contains a valid channel, range, or list (string) of DAQmx channels
        nchans is the number of channels to be read (integer)
        samplerate is the data sampling rate in Hz (integer)
        vrange is the voltage range limit (same for each channel, float)
    
    returns numpy array((n_samples. nchans), dtype=float64) of readings.
    '''    
    
    # Change these to describe DAQ measurement channel list and number of channels
    #chanlist = "Dev2/ai0:3"
    #chans = 4
        
    # DON'T change these - opens connection to DAQ.
    ai=Task()
    
    # Can change some of these if needed - sets up DAQ voltage readings - make sure 
    # you know what you're doping though!
    ai.CreateAIVoltageChan(chanlist, "", # DON'T change these, can change below:
                             DAQmx_Val_Diff, # 
                             #   DAQmx_Val_RSE, # Input coupling: Diff = V+ - V-, RSE = V+ - GND
                             -1.0*vrange, # -10.0 # Min voltage range: -1.0 or -10.0
                             1.0*vrange, # 10.0 # Max voltage range: 1.0 or 10.0 
                            DAQmx_Val_Volts, None) # DON'T change these
    
    
    # DON'T change any of this - configures DAQ timing & data arrays, 
    # starts the reading task, reads back the data, stops and clears the 
    # measurement task.
    N_SAMPLES = int(meas_duration * samplerate)
    ai.CfgSampClkTiming("", samplerate, DAQmx_Val_Rising, 
                          DAQmx_Val_FiniteSamps, N_SAMPLES) # DON'T change these 
    read = int32()
    SAMPLES=nchans*N_SAMPLES
    AIdata = np.zeros((N_SAMPLES,nchans), dtype=np.float64)
    ai.StartTask()
    ai.ReadAnalogF64(-1, 10.0, DAQmx_Val_GroupByScanNumber, 
                       AIdata, SAMPLES, byref(read), None)
    ai.StopTask()
    ai.ClearTask()
    
    
    # Print to terminal once readings done.
    #print('Read {} datapoints per channel'.format(read.value))
    
    return AIdata
 

   
