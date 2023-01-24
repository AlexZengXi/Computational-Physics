"""
Alex Zeng, 1007099373, Jan 20th 2023
Computational Lab - Section 1:
Fast Fourier Transform (FFT)
"""


from numpy import imag
from numpy.fft import fftfreq

save=True # if True then we save images as files

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

z1=abs(np.fft.fft(x))
z2=abs(np.fft.fft(y))
z3=abs(np.fft.fft(z))       # z3 peak at 10, 15 (checked manually)
# take the Fast Fourier Transforms of both x and y

# print("w1: max frequency is ", np.argmax(abs(z3[:100])))
# print("w2: max frequency is ", np.argmax(abs(z3[10:100])))
# a = max(abs(z3[10:100]))
# for i in range(len(z3[11:100])):
#     if a == (abs(z3[11:100]))[i]:
#         print(a)
#         print(i+10)
# print(z3[15])

"""
FFT It converts a signal into individual spectral components and 
thereby provides frequency information about the signal.

now go check out which "index is peaking"

then use fftfreq to findout which frequency that index represents 

QUESTIONS: what are the units of the fft diagram then?

"""

fig, ( (ax1,ax2), (ax3,ax4), (ax5,ax6) ) = plt.subplots(3,2,sharex='col',sharey='col')

# ploting the graphs
ax1.plot(time,x)
ax2.plot(z1)
ax3.plot(time,y)
ax4.plot(z2)
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
ax5.set_ylabel('Combining Both Waves')

mydpi=300
plt.tight_layout()
if (save): plt.savefig('section1_TwoWavesCombineWithFFT.png',dpi=mydpi)
plt.show()

############################################################################################
# finding the frequencies and relative amplitudes

# calculating the frequencies
n1 = len(z)
delta = 1
freq1 = fftfreq(n1, delta)      # Calculate frequencies of the transform in Hz
# freq are 0.05 and 0.075 in Hz
w = 2 * np.pi * freq1       # freq are 0.31416 and 0.47124 in rad/s

# calculating the amplitude
amp = 1 / N * z3



