"""
Alex Zeng, 1007099373, Jan 20th 2023
Computational Lab - Section 4
Studying the FFT of a sine function with time dependent frequency
"""



from numpy.fft import fftfreq
import matplotlib.pyplot as plt
import numpy as np

save=True # if True then we save images as files

N=100   # N is how many data points we will have in our sine wave
dt=100
time=np.linspace(0,N,dt)

A = 1
# w = t/4
y = A * np.sin(time * (time/400))

# calculating the FFT
y_fft = abs(np.fft.fft(y))

# ploting the graphs
fig, (ax1,ax2) = plt.subplots(1,2,sharex='col',sharey='col')
ax1.plot(time,y)
ax1.set_ylabel("y")
ax1.set_xlabel("Time [s]")
ax2.plot(y_fft)
ax2.set_ylabel("FFT(y)")
ax2.set_xlabel("Time [s]")
plt.show()

# saving the graph
mydpi=300
plt.tight_layout()
if(save): plt.savefig('section4_TimeVaryingSingleWaveAndFFT.png',dpi=mydpi)
plt.show()