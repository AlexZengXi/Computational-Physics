"""
Alex Zeng, 1007099373, Jan 20th 2023
Computational Lab - Section 3
Signal Filtering II
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftfreq

save=True # if True then we save images as files
mydpi=300

with open('noisy_sine_wave','rb') as file:
    data_from_file=pickle.load(file)
"""
the above few lines makes an array called data_from_file which contains
a noisy sine wave as long as you downloaded the file "noisy_sine_wave" 
and put it in the same directory as this python file

pickle is a Python package which nicely saves data to files. it can be
a little tricky when you save lots of data, but this file only has one
object (an array) saved so it is pretty easy
"""

import matplotlib.pyplot as plt

# plot 1
plt.plot(data_from_file)
xmax=300
plt.xlim(0,xmax)
plt.xlabel("Position-Time")
plt.ylabel("Pickle's Amplitude")
if (save): plt.savefig('section3.1_SingleWaveAndNoiseShort.png',dpi=mydpi)
plt.show()

number=len(data_from_file)
message="There are " + \
        str(number) + \
        " data points in total, only drawing the first " + \
        str(xmax)
print(message)

"""
The above section are for importing pickle's data to the code 
"""

N=2000   # N is how many data points we will have in our sine wave

time=np.arange(N)
data=data_from_file

data_fft=np.fft.fft(data) # take the Fast Fourier Transforms

# plot 2
plt.plot(time, data)
plt.xlabel('Time')
plt.ylabel('Position')
if (save): plt.savefig('section3.2_SingleWaveAndNoiseFullLength.png',dpi=mydpi)
plt.show()

# plot 3
plt.plot(abs(data_fft))
plt.xlabel('Absolute value of FFT of Position-Time\n(Frequency)')
plt.ylabel('Amplitude')
if (save): plt.savefig('section3.3_SingleWaveAndNoiseFullLengthWithFFT.png',dpi=mydpi)
plt.show()

# finding the max of noisy FFT
delta = 1/N     # Calculate frequencies of the transform in Hz
n1 = len(data)
freq1 = fftfreq(n1, delta)
w1 = 2 * np.pi * freq1      # Convert to angular frequencies

print("w1: max frequency is ", np.argmax(abs(data_fft[0:2000])),
      "with the amplitude being ",  max(abs(data_fft[0:2000])))
print("w1: max frequency is ", np.argmax(abs(data_fft[0:280])),
      "with the amplitude being ",  max(abs(data_fft[0:280])))
print("w1: max frequency is ", np.argmax(abs(data_fft[0:150])),
      "with the amplitude being ",  max(abs(data_fft[0:150])))
# check the python console for all the max

M=len(data_fft)       # length of x, with noise
freq=np.arange(M)  # frequency values, like time is the time values

width_1 = 0.01      # width=2*sigma**2 where sigma is the standard deviation
peak_1=286    # ideal value is approximately N/T1
filter_function_1=(np.exp(-(freq-peak_1)**2/width_1)+np.exp(-(freq+peak_1-M)**2/width_1))
z_filtered_1 = data_fft * filter_function_1

width_2 = 0.01      # width=2*sigma**2 where sigma is the standard deviation
peak_2 =154    # ideal value is approximately N/T1
filter_function_2=(np.exp(-(freq-peak_2)**2/width_2)+np.exp(-(freq+peak_2-M)**2/width_2))
z_filtered_2 = data_fft * filter_function_2

width_3 = 0.01      # width=2*sigma**2 where sigma is the standard deviation
peak_3=118    # ideal value is approximately N/T1
filter_function_3=(np.exp(-(freq-peak_3)**2/width_3)+np.exp(-(freq+peak_3-M)**2/width_3))
z_filtered_3 = data_fft * filter_function_3

filter_function = filter_function_1 + filter_function_2 + filter_function_3
z_filtered = z_filtered_1 + z_filtered_2 + z_filtered_3

"""
we choose Gaussian filter functions, fairly wide, with
one peak per spike in our FFT graph

we eyeballed the FFT graph to figure out decent values of 
peak and width for our filter function

a larger width value is more forgiving if your peak value
is slightly off

making width a smaller value, and fixing the value of peak,
will give us a better final result
"""

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# this gives us an array of 3 graphs, vertically aligned
ax1.plot(np.abs(data_fft))
ax2.plot(filter_function)
ax3.plot(np.abs(z_filtered))

"""
note that in general, the fft is a complex function, hence we plot
the absolute value of it. in our case, the fft is real, but the
result is both positive and negative, and the absolute value is still
easier to understand

if we plotted (abs(fft))**2, that would be called the power spectra
"""

fig.subplots_adjust(hspace=0)
ax1.set_ylim(0,13000)
ax2.set_ylim(0,1.2)
ax3.set_ylim(0,13000)
ax1.set_ylabel('Noisy FFT')
ax2.set_ylabel('Filter Function')
ax3.set_ylabel('Filtered FFT')
ax3.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')

plt.tight_layout()
if(save): plt.savefig('section3.4_FilteringProcess.png',dpi=mydpi)
plt.show()

cleaned=np.fft.ifft(z_filtered)
"""
ifft is the inverse FFT algorithm

it converts an fft graph back into a sinusoidal graph

we took the data, took the fft, used a filter function 
to eliminate most of the noise, then took the inverse fft
to get our "cleaned" version of the original data
"""


n1 = len(data_fft)
delta = 1
freq1 = fftfreq(n1, delta)      # Calculate frequencies of the transform in Hz

# calculating Theoretical graph
A1=2/N * np.abs(data_fft[286])   # wave amplitude
T1=1/freq1[286]  # wave period
y1=A1*np.sin(2.*np.pi*time/T1)

A2=2/N * np.abs(data_fft[154])
T2=1/freq1[154]
y2=A2*np.sin(2.*np.pi*time/T2)

A3=2/N * np.abs(data_fft[118])
T3=1/freq1[118]
y3=A3*np.sin(2.*np.pi*time/T3)

z = y1+y2+y3

# ploting
fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
ax1.plot(time,data)
ax2.plot(time,np.real(cleaned))
ax3.plot(time,z)
""" 
we plot the real part of our cleaned data - but since the 
original data was real, the result of our tinkering should 
be real so we don't lose anything by doing this

if you don't explicitly plot the real part, python will 
do it anyway and give you a warning message about only
plotting the real part of a complex number. so really, 
it's just getting rid of a pesky warning message
"""

fig.subplots_adjust(hspace=0)
ax1.set_xlim(0,300)
ax1.set_ylim(-30,30)
ax1.set_ylabel('Original Data')
ax2.set_ylabel('Filtered Data')
ax3.set_ylabel('Theoretical Data')
ax3.set_xlabel('Position-Time')

if(save): plt.savefig('section3.5_SingleWaveAndNoiseFFT.png',dpi=mydpi)
plt.show()
