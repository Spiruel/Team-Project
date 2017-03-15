import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools

from algorithms_1d import *
from Audio_waveform import waveform
from k_means_cluster import reconstruction_fn

# define a function to return the nearest value in a list
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
	
dir = "/Users/teodortzokov/Dropbox/TracerCo project team folder/"
dir1 = r'Large motor/'
test_name = 'large_30V_5mins.csv'
train_name = 'large_4V_nowater.csv'

training_data = np.loadtxt(dir + dir1 + train_name, delimiter=',', comments='#',skiprows=1, usecols=[0])
training_lowpass = Filters.movingaverage(training_data,20)
data = np.loadtxt(dir + dir1 + test_name, delimiter=',', comments='#',skiprows=1, usecols=[0])
data_lowpass = Filters.movingaverage(data,20)

#data_lowpass = np.c_[data_lowpass, np.zeros(len(data_lowpass)), np.zeros(len(data_lowpass))]
times, amps = waveform(data_lowpass)

marker = itertools.cycle(('^', '*', 'X', 'D', '+')) 

fig = plt.figure(figsize=(8,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1, 3])
ax0 = fig.add_subplot(gs[1])
ax1 = fig.add_subplot(gs[0], sharex=ax0)

ax0.plot(times, amps, 'b')
min, max = amps.min(), amps.max()
if np.abs(min) >= max:
	ax0.set_ylim([-np.abs(min)-0.1*np.abs(min), np.abs(min)+0.1*np.abs(min)])
else:
	ax0.set_ylim([-np.abs(max)-0.1*np.abs(max), np.abs(min)+0.1*np.abs(max)])
ax0.set_xlabel('Time / s', fontsize=14); ax0.set_ylabel('Amplitude / V', fontsize=14)

#anomaly plots here
displacement = 0
for test in [median_absolute_deviation, five_sigma, stddev_from_moving_average]:
	if test != stddev_from_moving_average:
		anom_indices = test(data_lowpass) #gets the anomaly indices on the data for that specific test #THIS IS RETURNING EMPTY ARRAY. WHY?
		print 'anom_indices:', anom_indices
	else:
		anom_indices = test(data) #gets the anomaly indices on the data for that specific test #THIS IS RETURNING EMPTY ARRAY. WHY?
		print 'anom_indices:', anom_indices
	
	ax1.plot(times[anom_indices], np.linspace(displacement,displacement,len(anom_indices)), '.', label=str(test.__name__), marker=marker.next())
	displacement += 1

print 'Doing histogram...'

difference = len(training_lowpass) - len(data_lowpass)
if difference > 0:
	training_lowpass = training_lowpass[:-difference]
elif difference < 0:
	training_lowpass = np.append(training_lowpass[:difference], training_lowpass)
	
bins = 100
ordered = sorted(training_lowpass)
div = len(ordered) // bins
chunks = [ ordered[int(round(div * i)): int(round(div * (i + 1)))] for i in range(bins) ]
meds = [np.median(chunk) for chunk in chunks]
reconstructed = [find_nearest(meds, val) for val in data_lowpass]
# print reconstructed
error = training_lowpass - reconstructed
histo_anomalies = np.array(np.where(np.abs(error) >= 5*np.std(error))[0])
print 'anom_indices', histo_anomalies
ax1.plot(times[histo_anomalies], np.linspace(displacement,displacement,len(histo_anomalies)), '.', label='histogram', marker=marker.next())
displacement += 1

print 'Doing k_means test...'
data_lowpass_norm = data_lowpass * (np.mean(np.abs(training_lowpass))/np.mean(np.abs(data_lowpass)))
reconstruction = reconstruction_fn(training_lowpass[:9000],data_lowpass_norm)
recon_error = reconstruction - data_lowpass_norm
K_anomalies = np.array(np.where(np.abs(recon_error) >= 5*np.std(recon_error))[0])
print 'anom_indices:', K_anomalies
ax1.plot(times[K_anomalies], np.linspace(displacement,displacement,len(K_anomalies)), '.', label='k_means', marker=marker.next())

ax1.legend(bbox_to_anchor=(0.01, -2.2), loc=2, borderaxespad=0., ncol=2)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_ylim([-1,5])
plt.gcf().subplots_adjust(hspace=.1)

plt.savefig('figures/comparison/'+str(test_name[:-4])+'_'+str(train_name[:-4])+'.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()