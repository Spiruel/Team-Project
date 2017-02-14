import pandas
import numpy as np
import scipy
import matplotlib.pyplot as plt

def median_absolute_deviation(data):
    """
    A timeseries is anomalous if the deviation of its latest datapoint with
    respect to the median is X times larger than the median of deviations.
    """
    data_median = np.median(data)
    demedianed = np.abs(data - data_median)
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


if __name__ == '__main__':

    data = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')[:,1]
    fs = 3000

    import Filters

    data_lowpass = Filters.movingaverage(data,50)
    
    median_absolute_deviation = median_absolute_deviation(data_lowpass)
    print median_absolute_deviation

    import Audio_waveform
    times = Audio_waveform.waveform(data_lowpass,fs)[0]

    anom_amplitudes = data_lowpass[median_absolute_deviation]
    anom_times = times[median_absolute_deviation]

    plt.plot(times,data)
    plt.plot(times,data_lowpass)
    plt.plot(anom_times,anom_amplitudes,'ro',markersize = 10)
    plt.show()
