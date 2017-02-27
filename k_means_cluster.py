from __future__ import division

import numpy as np
import scipy
import matplotlib.pyplot as plt
import Filters

from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=150) 

segment_len = 12
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
    plt.figure()
    n_graph_rows = 3
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


def reconstruction_fn(fitting_data, current_data, segment_length):
	'''fitting_data is the normal data taken from a healthy motor, in order to create your synthetic set of shapes described by the centroids.
	Current_data is then the data that you are trying to fit the synthetic reconstructed data to'''
	fitting_segments = sliding_chunker(fitting_data, segment_length=segment_len, slide_length=slide_len)
	windowed_segments = windowed_segments_fn(fitting_segments, segment_length=segment_len)

	centroids = cluster_centroids(windowed_segments)


	reconstruction = np.zeros(len(current_data))
	new_slide_len = int(segment_len/2)

	test_segments = sliding_chunker(current_data, segment_length=segment_len,slide_length=new_slide_len) 
	window = np.sin(np.linspace(0, np.pi, segment_len))**2

	centroids = clusterer.cluster_centers_ #take this outside for loop

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
	normal_data = Filters.movingaverage(np.loadtxt('data/12v_comparisontobaseline.csv', delimiter=',', comments='#')[:,0][0:8000], window_size=20)
	normal_data_av = np.mean(np.abs(normal_data))
	other_data = Filters.movingaverage(np.loadtxt('data/testinwater.csv', delimiter=',', comments='#')[:,0][16000:32000], window_size=20)
	other_data_av = np.mean(np.abs(other_data))
	other_data = other_data*(normal_data_av/other_data_av)
	################ other_data may need to be normalised to be the same average amplitude as normal_data ##################
	################ THIS IS VERY IMPORTANT!!!! ####################



	'''segments = sliding_chunker(normal_data, segment_length=segment_len, slide_length=slide_len)
	windowed_segments = windowed_segments_fn(segments, segment_length=segment_len)

	cluster_centroids = cluster_centroids(windowed_segments)



	reconstruction = np.zeros(len(normal_data))
	new_slide_len = int(segment_len/2)

	test_segments = sliding_chunker(normal_data, segment_length=segment_len,slide_length=new_slide_len) 
	window = np.sin(np.linspace(0, np.pi, segment_len))**2

	centroids = clusterer.cluster_centers_ #take this outside for loop

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

	reconstruction = reconstruction_fn(normal_data, other_data, segment_len)

	n_plot_samples = 3000

	error = reconstruction[0:n_plot_samples] - other_data[0:n_plot_samples]
	error_98th_percentile = np.percentile(error, 98)
	print("Maximum reconstruction error was %.5f" % error.max())
	print("98th percentile of reconstruction error was %.5f" % error_98th_percentile)

	plt.plot(other_data[0:n_plot_samples], label="Original")
	plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed")
	plt.plot(error[0:n_plot_samples], label="Reconstruction Error")
	plt.legend()
	plt.show()


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
	plt.show()'''

	#### plotting windowed segments and synthetic ones
	'''plot_waves(windowed_segments, step=3)
	plot_waves(cluster_centroids, step=3)'''
