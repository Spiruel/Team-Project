#run from main directory to quickly plot data that's been saved to csv.
#usage example: >python test_plotter.py testinwater or >python test_plotter.py testinwater.csv to plot data from data/testinwater.csv

import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
	print 'Incorrect arguments supplied.'
	sys.exit()
else:
	filename = 'data/' + str(sys.argv[1])
	if filename[-4:] != '.csv':
		filename += '.csv'

data = np.loadtxt(filename, delimiter=',', comments='#')
plt.plot(data[:,0], 'r'); plt.plot(data[:,1], 'g'); plt.plot(data[:,2], 'b')

split = 10

for i in range(0, 100, 1):
	split = 100 - i
	try:
		plt.plot(range(len(data[:,0]))[::int(len(data[:,0])/split)], np.array([np.std(i) for i in np.split(data[:,0], split)])-0.15, 'k--')
		plt.show()
	except:
		print 'passing', i
		continue

