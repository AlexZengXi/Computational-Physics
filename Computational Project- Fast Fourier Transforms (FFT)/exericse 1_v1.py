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
z3=np.fft.fft(z)
# take the Fast Fourier Transforms of both x and y

fig, ( (ax1,ax2), (ax3,ax4), (ax5,ax6) ) = plt.subplots(3,2,sharex='col',sharey='col')

# ploting the graphs
ax1.plot(time,x)
ax2.plot(np.abs(z1))
ax3.plot(time,y)
ax4.plot(np.abs(z2))
ax5.plot(time,z)
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

############################################################################################

# calculating the frequencies
n1 = len(x)
n2 = len(y)
n3 = len(z)

delta = 1
# Calculate frequencies of the transform in Hz
freq1 = fftfreq(n1, delta)
freq2 = fftfreq(n2, delta)
freq3 = fftfreq(n3, delta)
# Convert to angular frequencies, since the entire document is using it
w1 = 2 * np.pi * freq1
w2 = 2 * np.pi * freq2
w3 = 2 * np.pi * freq3

fig, ( (ax7,ax8), (ax9, ax10) ) = plt.subplots(2,2,sharex='col',sharey='col')

# ploting the graphs
ax7.plot(w1, imag(z1))
ax8.plot(w2, imag(z2))
ax9.plot(w3, imag(z3))

ax7.set_ylim(-600,600)
ax7.set_xlim(-200,200)
ax8.set_ylim(-100,100)
ax8.set_xlim(-200,200)

# remove the horizontal space between the top and bottom row
fig.subplots_adjust(hspace=0)

# labels
ax7.set_ylabel('Imag(wave 1) after FFT')
ax8.set_ylabel('Imag(wave 2) after FFT')
ax9.set_ylabel('Imag(both waves) after FFT')
ax7.set_xlabel('frequency of wave 1 [rad/s]')
ax8.set_xlabel('frequency of wave 2 [rad/s]')
ax9.set_xlabel('frequency of both waves [rad/s]')

"""
Failed attemped at finding max

# finding the max of noisy FFT
delta = 1/N     # Calculate frequencies of the transform in Hz
n1 = len(x)
freq1 = fftfreq(n1, delta)
w1 = 2 * np.pi * freq1      # Convert to angular frequencies
print("w1: max frequency is ", np.argmax(abs(z2)),
      "with the amplitude being ",  max(abs(z2)))
"""
#
# print("w1: max frequency is ", w1[np.argmax(imag(z1))],
#       "with the imaginary value being ", imag(z1).max())
# print("w2: max frequency is ", w2[np.argmax(imag(z2))],
#       "with the imaginary value being ", imag(z2).max())
# print("w3: max frequency is ", w3[np.argmax(imag(z3))],
#       "with the imaginary value being ", imag(z3).max())

mydpi=800
plt.tight_layout()

if (save): plt.savefig('ex1_TwoWavesCombineWithFFT_MaxFreq.png',dpi=mydpi)
plt.show()
