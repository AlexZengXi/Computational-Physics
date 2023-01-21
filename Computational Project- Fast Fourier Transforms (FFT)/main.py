# the Python implementation of FFT

import pylab
# import scipy as sp
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from scipy import *

tmin = 0
tmax = 2.4 * pi
delta = 0.2
t = arange(tmin, tmax, delta)
y = sin(2.5 * t)
Y = scipy.fft(y)

plt.figure(6)
plt.plot(t, y, 'bo')
plt.title('y(t) = sin(2.5 t)')
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.show()

# plt.figure(7)
# plt.plot(np.imag(Y))
# plt.title('Imaginary part of the Fourier transform of sin(2.5 t)')
# plt.xlabel('j')
# plt.ylabel("Yj")
# plt.show()