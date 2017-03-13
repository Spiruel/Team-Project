from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks
import scipy

if __name__ == '__main__':

	data = np.loadtxt('data/rusty_12V.csv', delimiter=',', comments='#',skiprows=1)

	chan1, chan2, chan3 = peaks.mult_channels(data)

	Lor_params = peaks.fit_peaks(chan1)
	print Lor_params