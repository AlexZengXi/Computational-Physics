# Computational Lab - Exercise 1
# Alex Zeng, 1007099373, Jan 20th 2023

save=False # if True then we save images as files

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
z3=np.fft.fft(z)
# take the Fast Fourier Transforms of both x and y

fig, ( (ax1,ax2), (ax3,ax4), (ax5,ax6) ) = plt.subplots(3,2,sharex='col',sharey='col')

# ploting the graphs
ax1.plot(time/N,x)
ax2.plot(np.abs(z1))
ax3.plot(time/N,y)
ax4.plot(np.abs(z2))
ax5.plot(time/N,z)
ax6.plot(np.abs(z3))

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

# calculating the frequencies
n1 = len(x)
n2 = len(y)
n3 = len(z)
# Calculate frequencies of the transform in Hz
freq1 = fftfreq(n1, delta)
# Convert to angular frequencies, since the entire document is using it
w = 2 * pi * freq