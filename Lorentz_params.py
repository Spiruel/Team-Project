from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks
import scipy

def split(data, sample_size = 9000):
	splitter = np.arange(0,len(data),sample_size)
	chunks = np.split(data,splitter,axis=0)[1:-1]
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

	data = np.loadtxt('data/large_20V.csv', delimiter=',', comments='#',skiprows=1)

	chan1, chan2, chan3 = peaks.mult_channels(data)

	healthy_amps, healthy_centres, healthy_widths, healthy_offsets =  healthy_params(chan1)
	print 'amplitude = ', np.mean(healthy_amps), ' +/- ', np.std(healthy_amps)
	print 'centres = ', np.mean(healthy_centres), ' +/- ', np.std(healthy_centres)
	print 'widths = ', np.mean(healthy_widths), ' +/- ', np.std(healthy_widths)

