import threading, sys
import numpy as np
from daqai import DAQmx_ReadAI as ReadAI

d1 = np.array([]); d2 = np.array([]); d3 = np.array([])

class DataCaptThread(threading.Thread):
    def run(self):
	DURATION = .05
	savename = None

	# Hardcoded values: these may need to be changed or obtained from commandline
	SAMPLE_RATE = 3000	# ADC input sample rate (Hz)

	AICHANNELS = "Dev1/ai0:2"	# Analog input channels
	NCHANS = 3			# number of analog input channels
	VMax = 10.0			# Maximum input voltage 
        try:
    		while True:
			data = ReadAI(DURATION, chanlist=AICHANNELS, nchans=NCHANS, samplerate=SAMPLE_RATE, vrange=VMax)

			d1 = np.concatenate((d1, data[:,0]))
			d2 = np.concatenate((d2, data[:,1]))
			d3 = np.concatenate((d3, data[:,2]))
	except KeyboardInterrupt:
	    	print 'Saving data and exiting!'
		filename = raw_input('Enter filename here: ')
		np.savetxt('data/'+filename+'.csv', np.c_[d1,d2,d3])
		sys.exit(1)

class GuiThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.root = Tk()
        self.lbl = Label(self.root, text="")

    def run(self):
        self.lbl(pack)
        self.lbl.after(1000, self.updateGUI)
        self.root.mainloop()

    def updateGUI(self):
        msg = "Data is True" if data else "Data is False"
        self.lbl["text"] = msg
        self.root.update()
        self.lbl.after(1000, self.updateGUI)

if __name__ == "__main__":
    DataCaptThread().start()
    GuiThread().start()

    try:
        while True:
            print 'oooooh'
    except KeyboardInterrupt:
        exit()
