from __future__ import division

import numpy as np
import scipy
import matplotlib.pyplot as plt
import Filters
from random import randint

from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=150) 

segment_len = 24
slide_len = 2

def sliding_chunker(input_data, segment_length, slide_length):
	'''Segments the input array into chunks of length segment_lenth samples. '''
	segments = []
	for start_pos in range(0, len(input_data), slide_length):
		end_pos = start_pos + segment_length
		# make a copy so changes to 'segments' doesn't modify the original ekg_data
		segment = np.copy(input_data[start_pos:end_pos])
		# if we're at the end and we've got a truncated segment, drop it
		if len(segment) != segment_length:
			continue
		segments.append(segment)
	return segments

def windowed_segments_fn(segments, segment_length):
	'''Applied a sine wave window to each segment, to ensure that each segment start and ends at zero and so avoid discontinuities 
	when combingin the learned segments. '''
	window_rads = np.linspace(0, np.pi, segment_length)
	window = np.sin(window_rads)**2
	windowed_segments = []
	for segment in segments:
		windowed_segment = np.copy(segment) * window
		windowed_segments.append(windowed_segment)
	return windowed_segments

def plot_waves(waves, step):
    """
    Plot a set of 9 waves from the given set, starting from the first one
    and increasing in index by 'step' for each subsequent graph
    """
    f = plt.figure(figsize=(8,4))
    n_graph_rows = 1
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            #axes.set_ylim([-0.05, 0.1])
            plt.plot(waves[wave_n])
            graph_n += 1
            wave_n += step
    # fix subplot sizes so that everything fits
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.style.use('seaborn-white')
    plt.show()

def cluster_centroids(windowed_segments):
	clusterer.fit(windowed_segments) #splits data into 150 clusters, so if there are 4000 windows, that makes 4000/150 windows in each cluster
	cluster_centroids = clusterer.cluster_centers_ #finds the centroid of each cluster, in n dimensional space, where n is equal to segment_len
	return cluster_centroids

def fit_centroids(data):
	'''returns centroids given an initial data set. Not currently in use for main code'''
	segments = sliding_chunker(data, segment_length=segment_len, slide_length=slide_len)
	windowed_segments = windowed_segments_fn(segments, segment_length=segment_len)
	centroids = cluster_centroids(windowed_segments)
	return centroids

def synthetic(training_data, segment_length):
	fitting_segments = sliding_chunker(training_data, segment_length=segment_len, slide_length=slide_len)
	windowed_segments = windowed_segments_fn(fitting_segments, segment_length=segment_len)

	centroids = cluster_centroids(windowed_segments)

	return centroids

def fitting(fitting_data, centroids, segment_length):

	reconstruction = np.zeros(len(fitting_data))
	new_slide_len = int(segment_length/2)

	test_segments = sliding_chunker(fitting_data, segment_length, slide_length=new_slide_len)

	window = np.sin(np.linspace(0, np.pi, segment_len))**2

	for segment_n, segment in enumerate(test_segments):
		# don't modify the data in segments
		segment = np.copy(segment)
		segment *= window
		nearest_centroid_idx = clusterer.predict(segment)[0]
		
		nearest_centroid = np.copy(centroids[nearest_centroid_idx])

		# overlay our reconstructed segments with an overlap of half a segment
		pos = segment_n * new_slide_len
		reconstruction[pos:pos+segment_len] += nearest_centroid
		#you add the begining of one centroid to the middle of the previous one. This is fine as the bigining of the new centroid
		#is zero and the previous centroid is maximum. Then as the previous centroid reduces in amplitude, the new one increases. ''' 
	return reconstruction


def reconstruction_fn(training_data, current_data, segment_length=24):
	'''training_data is the normal data taken from a healthy motor, in order to create your synthetic set of shapes described by the centroids.
	Current_data is then the data that you are trying to fit the synthetic reconstructed data to'''
	fitting_segments = sliding_chunker(training_data, segment_length=segment_len, slide_length=slide_len)
	windowed_segments = windowed_segments_fn(fitting_segments, segment_length=segment_len)

	centroids = cluster_centroids(windowed_segments)


	reconstruction = np.zeros(len(current_data))
	new_slide_len = int(segment_len/2)

	test_segments = sliding_chunker(current_data, segment_length=segment_len,slide_length=new_slide_len) 
	window = np.sin(np.linspace(0, np.pi, segment_len))**2

	for segment_n, segment in enumerate(test_segments):
		# don't modify the data in segments
		segment = np.copy(segment)
		segment *= window
		nearest_centroid_idx = clusterer.predict(segment)[0]
		
		nearest_centroid = np.copy(centroids[nearest_centroid_idx])

		# overlay our reconstructed segments with an overlap of half a segment
		pos = segment_n * new_slide_len
		reconstruction[pos:pos+segment_len] += nearest_centroid
		#you add the begining of one centroid to the middle of the previous one. This is fine as the bigining of the new centroid
		#is zero and the previous centroid is maximum. Then as the previous centroid reduces in amplitude, the new one increases. ''' 
	return reconstruction

if __name__ == '__main__':

	#testinwater = np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')
	normal_data = Filters.movingaverage(np.loadtxt('data/large_4V_nowater.csv', delimiter=',', comments='#',skiprows=1)[:,0][0:9000], window_size=20)
	normal_data_av = np.mean(np.abs(normal_data))
	other_data = Filters.movingaverage(np.loadtxt('data/large_4V_nowater.csv', delimiter=',', comments='#',skiprows=1)[:,0][9000:18000], window_size=20)
	other_data_av = np.mean(np.abs(other_data))
	other_data = other_data*(normal_data_av/other_data_av)
	#other_data = normal_data ########FOR TESTING
	################ other_data may need to be normalised to be the same average amplitude as normal_data ##################
	################ THIS IS VERY IMPORTANT!!!! ####################

	centroids = synthetic(normal_data, segment_len)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)

	# [line] = plt.plot([0],[0], label='error')
	# plt.ion()
	# dat = np.array([])
	# ax.set_ylim([-0.002,0.002])
	# ax.set_xlim([0,len(other_data)])
	
	# block = 500
	# for pos in range(0, len(other_data), block):
		# reconstruction = fitting(other_data[pos:pos+block], centroids, segment_len)
		# dat = np.append(dat, reconstruction)
		# line.set_xdata(range(len(dat)))
		# line.set_ydata(dat)
		# plt.pause(.05)
	
	# plt.ioff()
	
	# plt.plot(range(len(other_data)), other_data, label='orig data')
	# plt.legend(frameon=False)
	# plt.show()

	reconstruction = reconstruction_fn(normal_data, other_data, segment_len)

	n_plot_samples = 9000

	error = reconstruction[0:n_plot_samples] - other_data[0:n_plot_samples]
	error_98th_percentile = np.percentile(error, 98)
	print("Maximum reconstruction error was %.5f" % error.max())
	print("98th percentile of reconstruction error was %.5f" % error_98th_percentile)

	plt.style.use('seaborn-white')
	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False, figsize=(8,4))
	ax1.plot(other_data[0:n_plot_samples], 'b', lw=1, label="Original")
	ax2.plot(reconstruction[0:n_plot_samples], 'orange', lw=1, label="Reconstructed")
	ax3.plot(error[0:n_plot_samples], 'r', lw=1, label="Reconstruction Error")
	# Fine-tune figure; make subplots close to each other and hide x ticks for
	# all but bottom plot.
	f.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

	lim = np.max(np.abs(other_data))

	ax1.set_xlim(0,n_plot_samples)
	ax1.set_ylim([-lim-0.2*lim,lim+0.2*lim])
	ax2.set_ylim([-lim-0.2*lim,lim+0.2*lim])
	ax3.set_ylim([-lim-0.2*lim,lim+0.2*lim])
	for a in [ax1, ax2, ax3]:
		a.legend(frameon=False, loc='upper right')
	ax3.set_xlabel('Samples',fontsize=14)
	ax2.set_ylabel('Amplitude / V',fontsize=14)

	plt.style.use('seaborn-white')

	plt.savefig('figures/kmeans_large_4Vnowater.pdf', dpi=300, transparent=True, bbox_inches='tight')
	plt.savefig('figures/kmeans_large_4Vnowater.png', dpi=300, transparent=True, bbox_inches='tight')
	plt.show()

	
	distances = clusterer.inertia_
	print distances


	################ Plotting random windows 

	'''initial_segments = sliding_chunker(normal_data,segment_len,slide_len)
	windowed_segments = windowed_segments_fn(initial_segments,segment_len)
	norm_winds = windowed_segments

	clusters = cluster_centroids(windowed_segments)
	norm_clust =  clusters

	#plot_waves(windowed_segments,step=50)
	#plot_waves(clusters,step=3)

	f, axarr = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(10,4))
	axarr[0,0].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[0,1].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[0,2].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[0,3].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[0,4].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')

	axarr[1,0].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[1,1].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[1,2].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[1,3].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')
	axarr[1,4].plot(10**3*norm_winds[randint(0,len(norm_winds))], color='blue')

	plt.subplots_adjust(wspace=0.1, hspace=0.1)

	f.text(0.5,0.0,'Element in Segment',ha='center', fontsize = '14')
	f.text(0.07,0.5,'Amplitude / mV',va='center', fontsize = '14', rotation = 'vertical')

	plt.style.use('seaborn-white')

	plt.savefig('figures/kmeans_training.pdf', dpi=300, transparent=True, bbox_inches='tight')
	plt.savefig('figures/kmeans_training.png', dpi=300, transparent=True, bbox_inches='tight')

	#plt.show()

	#plt.clf()
	f, axarr = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(10,4))
	axarr[0,0].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[0,1].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[0,2].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[0,3].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[0,4].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')

	axarr[1,0].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[1,1].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[1,2].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[1,3].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')
	axarr[1,4].plot(10**3*norm_clust[randint(0,len(norm_clust))], color='orange')

	plt.subplots_adjust(wspace=0.1, hspace=0.1)

	f.text(0.5,0.0,'Element in Segment',ha='center', fontsize = '14')
	f.text(0.07,0.5,'Amplitude / mV',va='center', fontsize = '14', rotation = 'vertical')

	plt.style.use('seaborn-white')

	plt.savefig('figures/kmeans_synthetic.pdf', dpi=300, transparent=True, bbox_inches='tight')
	plt.savefig('figures/kmeans_synthetic.png', dpi=300, transparent=True, bbox_inches='tight')

	plt.show()'''

	##### Looking at a single segment and the quality of fit
	'''new_slide_len = int(segment_len/2)

	test_segments = sliding_chunker(normal_data, segment_length=segment_len,slide_length=new_slide_len) 
	print len(test_segments)
	#can make slide length half the length of the window length for the purpose of reconstructing the waveform.


	segment = np.copy(test_segments[500])
	# remember, the clustering was set up using the windowed data
	# so to find a match, we should also window our search key
	windowed_segment = segment * np.sin(np.linspace(0, np.pi, segment_len))**2
	# predict() returns a list of centres to cope with the possibility of multiple
	# samples being passed
	nearest_centroid_idx = clusterer.predict(windowed_segment)[0]
	nearest_centroid = np.copy(cluster_centroids[nearest_centroid_idx])
	plt.figure()
	plt.plot(segment, label="Original segment")
	plt.plot(windowed_segment, label="Windowed segment")
	plt.plot(nearest_centroid, label="Nearest centroid")
	plt.legend()
	plt.show()

	#### plotting windowed segments and synthetic ones
	plot_waves(windowed_segments, step=3)
	plot_waves(cluster_centroids, step=3)'''
