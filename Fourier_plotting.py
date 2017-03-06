from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks

data = np.loadtxt('data/gears_removed_baseline.csv', delimiter=',', comments='#')

chan1, chan2, chan3 = peaks.mult_channels(data)

print chan1

frequencies, sample_rate, amplitudes = peak.params(data)

print frequencies