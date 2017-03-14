# import modules and initialize plotting parameters
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'data/testinwater.csv', comments=['#','\"'], delimiter=',', usecols=0)

all_data = np.array(data)
# lets take half of the input data for training, and half for testing
# and create numpy arrays
half = int(len(data)/2)
train_data = all_data[:half]
test = all_data[half:]

#plt.figure(figsize=(16, 10))
plt.plot(train_data)
plt.xlabel('Amplitude /mV')
plt.ylabel('Count')
plt.show()

bins = 100
ordered = sorted(train_data)
div = len(ordered) // bins
chunks = [ ordered[int(round(div * i)): int(round(div * (i + 1)))] for i in range(bins) ]
meds = [np.median(chunk) for chunk in chunks]

n, bins, patches = plt.hist(ordered,100)
plt.grid()
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.show()

print('Minumum reading = {} mV., maximum = {} mV.'.format(np.amin(ordered), np.amax(ordered)))
print('Mean = {:0.2f} mV.'.format(np.mean(ordered)))
print('Standard deviation = {:0.2f} mV.'.format(np.std(ordered)))

# define a function to return the nearest value in a list
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
	
reconstructed = [find_nearest(meds, val) for val in test]
error = train_data - reconstructed

# plt.rc('text', usetex=True)  
# plt.rc('font', family='serif')  

plt.style.use('seaborn-white')
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(8,4))
ax1.plot(train_data, 'b', lw=1, label="Original")
ax2.plot(reconstructed, 'orange', lw=1, label="Reconstructed")
ax3.plot(error, 'r', lw=1, label="Reconstruction Error")
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	
# plt.ylim([-1-0.2,1+0.2])
for a in [ax1, ax2, ax3]:
	a.legend(frameon=False, loc='lower left')
	a.set_ylim([-0.16,0.16])
	
ax3.set_xlabel('Samples', fontsize=14)
ax2.set_ylabel('Amplitude / V', fontsize=14)
plt.savefig('figures/histogram.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

max_error = np.absolute(error).max()
error_99th_percentile = np.percentile(error, 99)
        
print('The maxiumum reconstruction error is: {:0.2f}'.format(max_error))
print('The 99th percentile of reconstruction error is: {:0.2f}'.format(error_99th_percentile))
