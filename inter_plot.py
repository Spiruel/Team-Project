import Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

fs=44100
duration = 5  # seconds
myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
print "Recording Audio"
sd.wait()
print "Audio recording complete , Play Audio"
sd.play(myrecording, fs)
sd.wait()
print "Play Audio Complete"

plt.plot(range(len(myrecording)),myrecording)
plt.show()

# f, Pxx_den = signal.periodogram(myrecording, 1000)
# plt.semilogy(f, Pxx_den)
# plt.ylim([1e-7, 1e2])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()

class App:
    def __init__(self, master):
        # Create a container
        frame = Tkinter.Frame(master)
        # Create 2 buttons
        self.button_left = Tkinter.Button(frame,text="< Decrease Slope",
                                        command=self.decrease)
        self.button_left.pack(side="left")
        self.button_right = Tkinter.Button(frame,text="Increase Slope >",
                                        command=self.increase)
        self.button_right.pack(side="left")

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig,master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def decrease(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y - 0.2 * x)
        self.canvas.draw()

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.canvas.draw()

root = Tkinter.Tk()
app = App(root)
root.mainloop()