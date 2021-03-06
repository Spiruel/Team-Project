import numpy as np
import lsanomaly
import scipy.signal
import matplotlib.pyplot as plt
        
def movingaverage(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')
	
# Use electrocardiogram data from PhysioNet as an example.
# More data can be downloaded from http://www.physionet.org/cgi-bin/ATM
# Select MIT-BIH Arrhythmia Database (mitdb), and export as CSV.
#X = np.loadtxt('data/MIT_physionet_sample.csv',skiprows=2, 
 #              usecols=(1,2), delimiter=',')
 
window_size = 100
			   
a = np.loadtxt(r'D:\Users\Samuel\Dropbox\TracerCo project team folder\12V motor\motornorm12V.csv', delimiter=',', usecols=[0])[:21600]
b = np.loadtxt(r'D:\Users\Samuel\Dropbox\TracerCo project team folder\12V motor\WD40_after paper_12V.csv', delimiter=',', usecols=[0])[:21600]
a = movingaverage(a, window_size)
b = movingaverage(b, window_size)
X = np.c_[a,b]

X[:,0] = X[:,0] - scipy.signal.medfilt(X[:,0],kernel_size=301)
X[:,1] = X[:,1] - scipy.signal.medfilt(X[:,1],kernel_size=301)

# Construct 4-D  observations from the original 2-D data: values at the
# current index and at a fixed lag behind.
N = X.shape[0]
lag = 10
X2 = np.zeros((N-lag,4))
for i in range(lag,N):
    X2[i-lag,0] = X[i,0]
    X2[i-lag,1] = X[i-lag,0]
    X2[i-lag,2] = X[i,1]
    X2[i-lag,3] = X[i-lag,1]   
    
X_train = X2[:5000,:]
X_test = X2[10000:20000,:]

# Train the model
anomalymodel = lsanomaly.LSAnomaly(rho=1, sigma=.05)
anomalymodel.fit(X_train)
#from sklearn import svm
#anomalymodel = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
#anomalymodel.fit(X_train)

# Predict anomalies statically (assuming iid samples)
y_pred_static = anomalymodel.predict_proba(X_test)

# Predict anomalies sequentially (assume known transition matrix and
# initial probabilities)
A = np.array([[.999, .001],[.01, .99]])
pi = np.array([.5,.5])
y_pred_dynamic = anomalymodel.predict_sequence(X_test, A, pi)
#help(anomalymodel.predict_sequence(X_test, A, pi))

plt.style.use('seaborn-white')
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(8,4))
ax1.plot(X_test[:,1], 'b', label='Train Data')
ax2.plot(X_test[:,3], 'orange', label='Test Data')
ax3.plot(y_pred_static[:,1],'r', label='Anomaly Score')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

for a in [ax1, ax2, ax3]:
	a.legend(frameon=False, loc='lower right')
	#a.set_ylim([-0.16,0.16])
	
ax3.set_xlabel('Samples', fontsize=14)
ax2.set_ylabel('Amplitude / V', fontsize=14)

# ax1 = plt.subplot(311)
# plt.plot(X_test[:,1], label='Train Data')
# plt.xticks(plt.xticks()[0], '', fontsize=8)
# plt.ylim(-0.003, 0.003)
# plt.yticks([-0.003, 0, 0.003])

# ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
# plt.plot(X_test[:,3], label='Test Data')     
# plt.xticks(plt.xticks()[0],'', fontsize=8)
# plt.ylim(-0.003, 0.003)
# plt.yticks([-0.003, 0, 0.003])


# ax3 = plt.subplot(313, sharex=ax1)
# plt.plot(y_pred_static[:,1],'r')
# plt.xticks(plt.xticks()[0],'', fontsize=8)
# plt.ylabel('Anomaly score\n(static)')
# plt.legend(frameon=True, loc='upper right')
# plt.yticks([0.0, 0.0002, 0.0004, 0.0006])

plt.savefig('figures/lsanomaly.pdf', dpi=300, transparent=True, bbox_inches='tight')
plt.show()