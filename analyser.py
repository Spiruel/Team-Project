import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('tp.csv', delimiter=',')
time, amplitude = data[:,0], data[:,1]

plt.plot(time, amplitude)
plt.show()