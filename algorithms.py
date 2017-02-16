from __future__ import division
import pandas
import numpy as np
import scipy
import matplotlib.pyplot as plt

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

    anomalies = np.where(normalised_median_deviation > 10)

    # Completely arbitary...triggers if the median deviation is
    # 6 times bigger than the median
    return anomalies


def grubbs(timereies):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    2 sided Grubbs test.
    """

    stdDev = np.std(timeseries)
    mean = np.mean(timeseries)
    z_score = np.abs(timeseries - mean) / stdDev #normalised residuals
    len_series = len(timeseries)
    threshold = scipy.stats.t.isf(0.05 / (2 * len_series), len_series - 2) 
    #upper critical values of the t distrubution with N - 2 degrees of freedo and a significance level of alpha/2N
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))
    #if any data point deviates from the mean by more than the Grubbs score, then it is classed as an outlier. 

    anomalies = np.where(z_score > grubbs_score)

    return anomalies

def five_sigma(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints
    are outside of five standard deviations of the mean.
    """
    data = np.abs(timeseries)
    mean = np.mean(timeseries)
    print mean
    stdDev = np.std(timeseries)

    norm_resids = np.abs(timeseries - mean) / stdDev

    anomalies = np.where(norm_resids > 5)[0]

    return anomalies


def stddev_from_moving_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the moving average. This is better for finding anomalies with
    respect to the short term trends.
    """
    timeseries_av = Filters.movingaverage(timeseries,50)
    
    diff = np.abs(timeseries - timeseries_av)

    #series = pandas.Series(timeseries)
    #expAverage = pandas.stats.moments.ewma(series, com=50)
    #stdDev = pandas.stats.moments.ewmstd(series, com=50)



    return abs(series.iget(-1) - expAverage.iget(-1)) > 2 * stdDev.iget(-1)


if __name__ == '__main__':

    data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')[:,1]
    fs = 3000

    import Filters

    data_lowpass = Filters.movingaverage(data,50)
    
    anomaly_indices = five_sigma(data_lowpass)
    print anomaly_indices

    import Audio_waveform
    times = Audio_waveform.waveform(data_lowpass,fs)[0]

    anom_amplitudes = data_lowpass[anomaly_indices]
    anom_times = times[anomaly_indices]

    plt.plot(times,data)
    plt.plot(times,data_lowpass)
    plt.plot(anom_times,anom_amplitudes,'ro',markersize = 10)
    plt.show()
