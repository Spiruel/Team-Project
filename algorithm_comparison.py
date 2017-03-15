import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from algorithms_1d import *
from Audio_waveform import waveform
from k_means_cluster import reconstruction_fn

training_data = np.loadtxt('data/large_4V_nowater.csv', delimiter=',', comments='#',skiprows=1)[:,0][:9000]
training_lowpass = Filters.movingaverage(training_data,20)
data = np.loadtxt('data/large_30V_5mins.csv', delimiter=',', comments='#',skiprows=1)[:,0]
data_lowpass = Filters.movingaverage(data,20)

#data_lowpass = np.c_[data_lowpass, np.zeros(len(data_lowpass)), np.zeros(len(data_lowpass))]
times, amps = waveform(data_lowpass)

fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1, 3])
ax0 = fig.add_subplot(gs[1])
ax1 = fig.add_subplot(gs[0], sharex=ax0)

ax0.plot(times, amps, 'b')
ax0.set_xlabel('Time / s', fontsize=14); ax0.set_ylabel('Amplitude / V', fontsize=14)

#anomaly plots here
displacement = 0
for test in [median_absolute_deviation, five_sigma, stddev_from_moving_average]:
	if test != stddev_from_moving_average:
		anom_indices = test(data_lowpass) #gets the anomaly indices on the data for that specific test #THIS IS RETURNING EMPTY ARRAY. WHY?
		print 'anom_indices:', anom_indices
	else:
		anom_indices = test(data)[0] #gets the anomaly indices on the data for that specific test #THIS IS RETURNING EMPTY ARRAY. WHY?
		print 'anom_indices:', anom_indices
	
	ax1.plot(times[anom_indices], np.linspace(displacement,displacement,len(anom_indices)), '.', label=str(test.__name__))
	displacement += 1

reconstruction = reconstruction_fn(training_lowpass,data_lowpass)
recon_error = reconstruction - data_lowpass

	
#load anomaly indices of same dataset from kmeans where numpy.where(recon_error >= 3*np.std(recon_error))
#plot anomaly indices on ax1 as before
#anom_indices = np.loadtxt(r'kmeans_anomalies.csv', comments=['#','\"'], delimiter=',')
#ax1.plot(times[anom_indices], np.linspace(displacement,displacement,len(anom_indices)), '.', label='k means')

ax1.legend(frameon=False)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_ylim([-1,5])
plt.gcf().subplots_adjust(hspace=.1)

# plt.savefig('figures/algo_comparison.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()