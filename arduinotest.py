#!/usr/bin/python
 
import time
import serial
import matplotlib.pyplot as plt
 
max_valid_read = 1023
reference_voltage = 3.3
update_interval = 2e-4
 
ser = serial.Serial(
 port = 'COM5',
 baudrate = 115200,
 )
 
time.sleep(1.5)
ser.flush()
ser.write('2')
 
r = 0
r_time = 0
analog_reads = [];
analog_times = [];
while r <= max_valid_read:
 b1 = ord(ser.read(1))
 b2 = ord(ser.read(1))
 r = b1 + b2*256
 if(r <= max_valid_read):
   analog_reads.append((r*reference_voltage)/max_valid_read)
   analog_times.append(r_time)
   r_time = r_time + update_interval
 
plt.plot(analog_times, analog_reads)
plt.show()