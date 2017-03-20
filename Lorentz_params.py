from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks
import scipy

def split(data, sample_size = 3000):
	splitter = np.arange(0,len(data),sample_size)
	chunks = np.split(data,splitter,axis=0)[1:-1]
	return chunks

def gauss_params(gauss_data):
	chunks = split(gauss_data)
	amps = np.array([])
	centres = np.array([])
	widths = np.array([])
	offsets = np.array([])
	for chunk in chunks:
		gauss_param = peaks.fit_peaks(chunk)[0]
		amp, centre, width, offset = gauss_param
		amps = np.append(amps, amp)
		centres = np.append(centres,centre)
		widths = np.append(widths, width)
		offsets = np.append(offsets,offset)
	
	return np.array([amps,centres,widths,offsets])



if __name__ == '__main__':

	file_name = 'gears_removed_snipped1'

	data = np.loadtxt('/Users/teodortzokov/Dropbox/TracerCo project team folder/Gears removed/'+file_name+'.csv', delimiter=',', comments='#',skiprows=1)

	chan1, chan2, chan3 = peaks.mult_channels(data)

	gauss_amps, gauss_centres, gauss_widths, gauss_offsets =  gauss_params(chan1)
	print 'amplitude = ', np.mean(gauss_amps), ' +/- ', np.std(gauss_amps)/np.sqrt(len(gauss_amps))
	print 'centres = ', np.mean(gauss_centres), ' +/- ', np.std(gauss_centres)/np.sqrt(len(gauss_centres))
	print 'widths = ', np.mean(gauss_widths), ' +/- ', np.std(gauss_widths)/np.sqrt(len(gauss_widths))

	gauss_amps, gauss_centres, gauss_widths, gauss_offsets =  gauss_params(chan2)
	print 'amplitude = ', np.mean(gauss_amps), ' +/- ', np.std(gauss_amps)/np.sqrt(len(gauss_amps))
	print 'centres = ', np.mean(gauss_centres), ' +/- ', np.std(gauss_centres)/np.sqrt(len(gauss_centres))
	print 'widths = ', np.mean(gauss_widths), ' +/- ', np.std(gauss_widths)/np.sqrt(len(gauss_widths))

	gauss_amps, gauss_centres, gauss_widths, gauss_offsets =  gauss_params(chan3)
	print 'amplitude = ', np.mean(gauss_amps), ' +/- ', np.std(gauss_amps)/np.sqrt(len(gauss_amps))
	print 'centres = ', np.mean(gauss_centres), ' +/- ', np.std(gauss_centres)/np.sqrt(len(gauss_centres))
	print 'widths = ', np.mean(gauss_widths), ' +/- ', np.std(gauss_widths)/np.sqrt(len(gauss_widths))

