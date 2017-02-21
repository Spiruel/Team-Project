from __future__ import division
import pandas
import numpy as np
import scipy
import matplotlib.pyplot as plt
import Filters
import Audio_waveform

def median_absolute_deviation(timeseries):
	"""
	A timeseries is anomalous if the deviation of its latest datapoint with
	respect to the median is X times larger than the median of deviations.
	"""
	data_median = np.median(timeseries)
	demedianed = np.abs(timeseries - data_median)
	median_deviation = np.median(demedianed)

	if median_deviation == 0:
		return False

	normalised_median_deviation = demedianed / median_deviation

	# The test statistic is infinite when the median is zero,
	# so it becomes super sensitive. We play it safe and skip when this happens.

	anomalies = np.where(normalised_median_deviation > 6)[0]

	# Completely arbitary...triggers if the median deviation is
	# 6 times bigger than the median
	return anomalies

if __name__ == '__main__':

    data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')
    fs = 3000


    data_lowpass = Filters.movingaverage(data,50)
    
    anomaly_indices = median_absolute_deviation(data_lowpass)
    print anomaly_indices