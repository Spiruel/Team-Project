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
plt.plot(data)
plt.show()