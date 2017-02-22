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
	data_median = np.median(timeseries, axis=0)
	demedianed = np.abs(timeseries - data_median)
	median_deviation = np.median(demedianed, axis=0)
	

	#if median_deviation == 0:
	#	return False

	normalised_median_deviation = demedianed / median_deviation

	# The test statistic is infinite when the median is zero,
	# so it becomes super sensitive. We play it safe and skip when this happens.

	#anomalies = np.where(normalised_median_deviation > 6)[0]
	anomalies = np.array([np.where(column > 6)[0] for column in normalised_median_deviation.T])
	# Completely arbitary...triggers if the median deviation is
	# 6 times bigger than the median
	return anomalies

def grubbs(timeseries):
	"""
	A timeseries is anomalous if the Z score is greater than the Grubb's score.
	2 sided Grubbs test.
	"""

	stdDev = np.std(timeseries, axis=0)
	mean = np.mean(timeseries, axis=0)
	z_score = np.abs(timeseries - mean) / stdDev #normalised residuals
	len_series = timeseries.shape[0]
	threshold = scipy.stats.t.isf(0.05 / (2 * len_series), len_series - 2) 
	#upper critical values of the t distribution with N - 2 degrees of freedo and a significance level of alpha/2N
	threshold_squared = threshold ** 2
	grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))
	#if any data point deviates from the mean by more than the Grubbs score, then it is classed as an outlier. 

	#anomalies = np.where(z_score[:,0] > grubbs_score)[0]
	anomalies = np.array([np.where(column > grubbs_score)[0] for column in z_score.T])

	return anomalies

def five_sigma(timeseries):
	"""
	A timeseries is anomalous if the average of the last three datapoints
	are outside of five standard deviations of the mean.
	"""
	data = np.abs(timeseries)
	mean = np.mean(timeseries, axis=0)
	stdDev = np.std(timeseries, axis=0)

	norm_resids = np.abs(timeseries - mean) / stdDev

	#anomalies = np.where(norm_resids > 5)[0]
	anomalies = np.array([np.where(column > 5)[0] for column in norm_resids.T])
	
	return anomalies


def stddev_from_moving_average(timeseries):
	"""
	A timeseries is anomalous if the absolute value of the average of the latest
	three datapoint minus the moving average is greater than three standard
	deviations of the moving average. This is better for finding anomalies with
	respect to the short term trends.
	"""

	series = pandas.DataFrame(timeseries)
	expAverage = series.rolling(window=50,center=False).mean()
	stdDev = series.rolling(window=50,center=False).std()

	indices_bool = np.abs(series - expAverage) > 4 * stdDev

	# indices = np.where(indices_bool)
	indices = np.array([np.where(column)[0] for column in indices_bool.T])


	return indices


if __name__ == '__main__':

    data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')[:,1]
    fs = 3000


    data_lowpass = Filters.movingaverage(data,50)
    
    anomaly_indices = grubbs(data_lowpass)
    print anomaly_indices

    times = Audio_waveform.waveform(data_lowpass,fs)[0]

    anom_amplitudes = data_lowpass[anomaly_indices]
    anom_times = times[anomaly_indices]

    plt.plot(times,data)
    plt.plot(times,data_lowpass)
    plt.plot(anom_times,anom_amplitudes,'ro',markersize = 10)

    plt.xlabel('Time/s')
    plt.ylabel('Amplitude')
    plt.show()
