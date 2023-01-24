"""
Alex Zeng, 1007099373, Jan 20th 2023
Computational Lab - Section 4
Studying the FFT of a sine function with time dependent frequency
"""

from numpy.fft import fftfreq
import matplotlib.pyplot as plt
import numpy as np

save=True # if True then we save images as files
mydpi=300
plt.tight_layout()

N=100   # N is how many data points we will have in our sine wave
dt=100
time=np.linspace(0,N,dt)

A = 1
# w = t/4
y = A * np.sin(time * (time/400))

# calculating the FFT
y_fft = abs(np.fft.fft(y))

# saving the graphs
plt.plot(time,y)
plt.ylabel("y")
plt.xlabel("Time [s]")
plt.show()
if(save): plt.savefig('section4.1_TimeVaryingSingleWavePT.png',dpi=mydpi)

plt.plot(y_fft)
plt.ylabel("FFT(y)")
plt.xlabel("Time [s]")
plt.show()
if(save): plt.savefig('section4.2_TimeVaryingSingleWaveWithFFT.png',dpi=mydpi)


