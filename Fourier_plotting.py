from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import n_peaks_lorentzian as peaks
import scipy

if __name__ == '__main__':

	data = np.loadtxt('data/gears_removed_baseline.csv', delimiter=',', comments='#')

	chan1, chan2, chan3 = peaks.mult_channels(data)

	frequencies1, sample_rate1, amplitudes1 = peaks.params(chan1)
	peak_indices1, peak_freqs1, peak_amplitudes1 = peaks.find_peaks(chan1)


	frequencies2, sample_rate2, amplitudes2 = peaks.params(chan2)
	peak_indices2, peak_freqs2, peak_amplitudes2 = peaks.find_peaks(chan2)

	frequencies3, sample_rate3, amplitudes3 = peaks.params(chan3)
	peak_indices3, peak_freqs3, peak_amplitudes3 = peaks.find_peaks(chan3)


	#### plotting

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
	ax1.semilogy(frequencies1, amplitudes1, color ='blue')
	ax1.semilogy(peak_freqs1, peak_amplitudes1, 'ko', markersize=10)

	ax2.semilogy(frequencies2, amplitudes2, color='red')
	ax2.semilogy(peak_freqs2, peak_amplitudes2, 'ko', markersize=10)


	ax3.semilogy(frequencies3, amplitudes3, color='green')
	ax3.semilogy(peak_freqs3, peak_amplitudes3, 'ko', markersize=10)

	ax1.set_ylabel('$|FFT|^2$ for x-axis')
	ax2.set_ylabel('$|FFT|^2$ for y-axis')
	ax3.set_ylabel('$|FFT|^2$ for z-axis')
	# Fine-tune figure; make subplots close to each other and hide x ticks for
	# all but bottom plot.
	f.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	ax3.set_xlim(0,1500)
	plt.xlabel('Frequency / Hz')

	'''left, bottom, width, height = [0.6, 0.43, 0.25, 0.15]
	ax11 = f.add_axes([left, bottom, width, height])

	ax11.set_xlim(0,250)
	ax11.set_ylim(0,np.max(amplitudes2)/20)
	ax11.semilogy(frequencies2,amplitudes2, color='red')'''

	plt.style.use('seaborn-white')


	plt.show()

	#plt.savefig(, dpi=300, transparent=True, bbox_inches='tight')