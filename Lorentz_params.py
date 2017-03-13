from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks
import scipy

def split(data, sample_size = 9000):
	splitter = np.arange(0,len(data),sample_size)
	chunks = np.split(data,splitter,axis=0)[1:-2]
	return chunks

def healthy_params(healthy_data):
	chunks = split(healthy_data)
	amps = np.array([])
	centres = np.array([])
	widths = np.array([])
	offsets = np.array([])
	for chunk in chunks:
		healthy_param = peaks.fit_peaks(chunk)[0]
		amp, centre, width, offset = healthy_param
		amps = np.append(amps, amp)
		centres = np.append(centres,centre)
		widths = np.append(widths, width)
		offsets = np.append(offsets,offset)
	
	return np.array([amps,centres,widths,offsets])



if __name__ == '__main__':

	#slice up healthy motor data into segments of about 3 secs

	#get lorentzian parameters for each segment - create a database of healthy motor parameters

	#slice unhealthy motor data into 3 second chunks and find lorentzian parameters for each. 

	#add each unhealthy set of paramaters to dataase of healthy parameters and perform simple standard deviation test to see if A, HWHM or centre deviate significantly.

	data = np.loadtxt('data/small_misaligned_2.csv', delimiter=',', comments='#',skiprows=1)

	chan1, chan2, chan3 = peaks.mult_channels(data)

	healthy_amps, healthy_centres, healthy_widths, healthy_offsets =  healthy_params(chan1)
	
