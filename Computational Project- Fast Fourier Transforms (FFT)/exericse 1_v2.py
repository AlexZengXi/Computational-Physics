# Computational Lab - Exercise 1
# Alex Zeng, 1007099373, Jan 20th 2023
from numpy import imag
from numpy.fft import fftfreq

save=True # if True then we save images as files

from random import gauss
import matplotlib.pyplot as plt
import numpy as np

N=200   # N is how many data points we will have in our sine wave

time=np.arange(N)

A1=5.   # wave amplitude
T1=20.  # wave period
y1=A1*np.sin(2.*np.pi*time/T1)

A2=3.
T2=13.
y2=A2*np.sin(2.*np.pi*time/T2)

x = y1      # sine wave 1
y = y2      # sine wave 2
z = y1 + y2     # combiniation of sine wave 1 and 2

z1=np.fft.fft(x)
z2=np.fft.fft(y)
z3=abs(np.fft.fft(z))
# z3 peak at 10, 15
# take the Fast Fourier Transforms of both x and y
"""
once you take the fft, you can see 
"""

fig, ( (ax1,ax2), (ax3,ax4), (ax5,ax6) ) = plt.subplots(3,2,sharex='col',sharey='col')

# ploting the graphs
ax1.plot(time,x)
ax2.plot(np.abs(z1))

ax3.plot(time,y)
ax4.plot(np.abs(z2))

ax5.plot(time,z)
ax6.plot(z3)

# remove the horizontal space between the top and bottom row
fig.subplots_adjust(hspace=0)

# labels
ax5.set_xlabel('Position-Time')
ax6.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
ax3.set_ylim(-13,13)
ax4.set_ylim(0,600)
ax1.set_ylabel('Wave #1')
ax3.set_ylabel('Wave #2')
ax5.set_ylabel('Combining both waves')

mydpi=300
plt.tight_layout()

if (save): plt.savefig('ex1_TwoWavesCombineWithFFT.png',dpi=mydpi)
plt.show()

############################################################################################

# calculating the frequencies
n1 = len(z)

delta = 1
# Calculate frequencies of the transform in Hz
freq1 = fftfreq(n1, delta)
# freq are 0.05 and 0.075 in Hz
w = 2 * np.pi * freq1
# freq are 0.31416 and 0.47124 in rad/s
