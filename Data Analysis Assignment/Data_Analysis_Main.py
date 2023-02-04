import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 12}
rc('font', **font)
# This changes the fonts for all graphs to make them bigger.

save=False # if True then we save images as files
mydpi=300

def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
# This is my fitting function, a Guassian with a uniform background.

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall)) # filter func
    yy[:1000]=0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)
# fit_pulse can be used by curve_fit to fit a pulse to the pulse_shape


with open("calibration_p3.pkl","rb") as file:
    calibration_data=pickle.load(file)
with open("noise_p3.pkl","rb") as file:
    noise_data=pickle.load(file)
with open("signal_p3.pkl","rb") as file:
    signal_data=pickle.load(file)


pulse_template = pulse_shape(20,80)
# plt.plot(pulse_template/2000, label='Pulse Template', color='r')
# for itrace in range(10):
#     plt.plot(calibration_data['evt_%i'%itrace], alpha=0.3)
# plt.xlabel('Sample Index')
# plt.ylabel('Readout (V)')
# plt.title('Calibration data (10 sets)')
# plt.legend(loc=1)
# plt.show()
""" 
This shows the first 10 data sets on top of each other.
Always a good idea to look at some of your data before analysing it!
It also plots our pulse template which has been scaled to be slightly 
larger than any of the actual pulses to make it visible.
"""
amp1=np.zeros(1000)     # max-min
amp2=np.zeros(1000)     # max-baseline
area1=np.zeros(1000)    # integral of pre-pulse
area2=np.zeros(1000)    # integral of the pulse
area3=np.zeros(1000)    # integral of after-pulse
pulse_fit=np.zeros(1000)
amps = [amp1, amp2, area1, area2, area3, pulse_fit]
"""
These are the 6 energy estimators as empty arrays of the correct size.
"""

stds = []
for i in range(len(noise_data)):
    stds.append(np.std(noise_data['evt_%i'%i]))
std = np.mean(stds)
sig = np.full(len(noise_data['evt_0']), std)
sig = np.where(sig==0, 1, sig)
"""
calculating the std of the noise
"""

xx=np.linspace(0, 4095, 4096)
for ievt in range(1000):
    current_data = calibration_data['evt_%i'%ievt]

    amp1[ievt] = np.max(current_data) - np.min(current_data)
    # max-baseline, where baseline: average of the pre-pulse region
    amp2[ievt] = np.max(current_data) - np.average(current_data[0:1000])
    area1[ievt] = np.sum(current_data[0:1000])
    area2[ievt] = np.sum(current_data) - np.average(current_data[0:1000])
    area3[ievt] = np.sum(current_data[1000:1100])
    popt, pcov = curve_fit(fit_pulse, xx, current_data,
                           sigma=sig, absolute_sigma=True)
    pulse_fit[ievt] = popt[0]
"""
Calculating all amplitude estimators.
"""

# converting from V to mV
for amp in amps:
    amp *= 1000

def plotting(num_bins1, bin_range1, amps, p, title):
    n1, bin_edges1, _ = plt.hist(amps, bins=num_bins1, range=bin_range1,
                                 histtype='step', label='Data')
    # This plots the histogram AND saves the counts and bin_edges for later use

    plt.xlabel('Maximum Value (mV)')
    plt.ylabel('Events / %2.2f mV' % ((bin_range1[-1] - bin_range1[0]) / num_bins1));
    plt.xlim(bin_range1)
    # If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

    bin_centers1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
    """
    This gives us the x-data which are the centres of each bin.
    This is visually better for plotting errorbars.
    More important, it's the correct thing to do for fitting the
    Gaussian to our histogram.
    It also fixes the shape -- len(n1) < len(bin_edges1) so we
    cannot use 
    plt.plot(n1, bin_edges1)
    as it will give us a shape error.
    """

    sig1 = np.sqrt(n1)
    sig1 = np.where(sig1 == 0, 1, sig1)
    # The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

    plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
    # This adds errorbars to the histograms, where each uncertainty is sqrt(y)

    popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1,
                             sigma=sig1, p0=p, absolute_sigma=True)
    n1_fit = myGauss(bin_centers1, *popt1)
    """
    n1_fit is our best fit line using our data points.
    Note that if you have few enough bins, this best fit
    line will have visible bends which look bad, so you
    should not plot n1_fit directly. See below.
    """

    chisquared1 = np.sum(((n1 - n1_fit) / sig1) ** 2)
    dof1 = num_bins1 - len(popt1)
    # Number of degrees of freedom is the number of data points less the number of fitted parameters

    x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y_bestfit1 = myGauss(x_bestfit1, *popt1)
    # Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 10 data points!

    fontsize = 12
    plt.plot(x_bestfit1, y_bestfit1, label='Fit')
    plt.title('original '+ title + ' Amplitude Estimator')
    # plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
    # plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
    # plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    # plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
    # plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)


    # find smallest sigma
    # print('o_', title, ' chi2: ', "{:.4f}".format(1-chi2.cdf(chisquared1,dof1)),
    #       '   sigma: ', "{:00.4f}".format(popt1[2]),
    #       '   mu: ', "{:00.4f}".format(popt1[1]))
    # sigmal: std (around 90%), mu: mean, chi2 (around 1)

    plt.legend(loc=1)
    plt.show()

    return popt1

# graph 1
num_bins_1=60
bin_range_1=(0.25, 0.4)
p_1=(60, 0.31, 0.07, 5)        # myGauss(x, A, mean, width, base)
# plotting(num_bins_1, bin_range_1, amp1, p_1, 'amp 1')

# graph 2
num_bins_2=45
bin_range_2=(0.16, 0.33)
p_2=(200, 0.23, 0.1, 7)        # myGauss(x, A, mean, width, base)
# plotting(num_bins_2, bin_range_2, amp2, p_2, 'amp 2')

# graph 3
num_bins_3=40
bin_range_3=(-100, 100)
p_3=(200,0,100,5)        # myGauss(x, A, mean, width, base)
# plotting(num_bins_3, bin_range_3, area1, p_3, 'amp 3')

# graph 4
num_bins_4=40
bin_range_4=(-200, 200)
p_4=(100,50,150,0)        # myGauss(x, A, mean, width, base)
# plotting(num_bins_4, bin_range_4, area2, p_4, 'amp 4')

# graph 5
num_bins_5=80
bin_range_5=(-50, 80)
p_5=(300,21,50,0)        # myGauss(x, A, mean, width, base)
# conv_5 = plotting(num_bins_5, bin_range_5, area3, p_5, 'amp 5')

# graph 6
num_bins_6=80
bin_range_6=(-5, 5)
p_6=(800,0,1,0)        # myGauss(x, A, mean, width, base)
# plotting(num_bins_6, bin_range_6, pulse_fit, p_6, 'amp 6')

"""
Now your task is to find the calibration factor which converts the
x-axis of this histogram from mV to keV such that the peak (mu) is 
by definition at 10 keV. You do this by scaling each estimator (i.e.
the values of amp1) by a multiplicative constant with units mV / keV.
Something like:

energy_amp1 = amp1 * conversion_factor1

where you have to find the conversion_factor1 value. Then replot and
refit the histogram using energy_amp1 instead of amp1. 
If you do it correctly, the new mu value will be 10 keV, and the new 
sigma value will be the energy resolution of this energy estimator.

Note: you should show this before/after conversion for your first
energy estimator. To save space, only show the after histograms for
the remaining 5 energy estimators.
"""

def plotting_with_calibration(popt1, num_bins1, bin_range1, amps, p, title, count):

    conv = 10 / popt1[1]  # conversion factor
    amps *= conv

    n1, bin_edges1, _ = plt.hist(amps, bins=num_bins1, range=bin_range1,
                                 histtype='step', label='Data')
    # This plots the histogram AND saves the counts and bin_edges for later use

    plt.xlabel('Energy Estimator: Maximum Value (keV)')
    plt.ylabel('Events / %2.2f keV' % ((bin_range1[-1] - bin_range1[0]) / num_bins1));
    plt.xlim(bin_range1)
    # If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

    bin_centers1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
    """
    This gives us the x-data which are the centres of each bin.
    This is visually better for plotting errorbars.
    More important, it's the correct thing to do for fitting the
    Gaussian to our histogram.
    It also fixes the shape -- len(n1) < len(bin_edges1) so we
    cannot use 
    plt.plot(n1, bin_edges1)
    as it will give us a shape error.
    """

    sig1 = np.sqrt(n1)
    sig1 = np.where(sig1 == 0, 1, sig1)
    # The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

    plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
    # This adds errorbars to the histograms, where each uncertainty is sqrt(y)

    popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1,
                             sigma=sig1, p0=p, absolute_sigma=True)
    n1_fit = myGauss(bin_centers1, *popt1)
    """
    n1_fit is our best fit line using our data points.
    Note that if you have few enough bins, this best fit
    line will have visible bends which look bad, so you
    should not plot n1_fit directly. See below.
    """

    chisquared1 = np.sum(((n1 - n1_fit) / sig1) ** 2)
    dof1 = num_bins1 - len(popt1)
    # Number of degrees of freedom is the number of data points less the number of fitted parameters

    x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y_bestfit1 = myGauss(x_bestfit1, *popt1)
    # Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 10 data points!

    fontsize = 12
    plt.plot(x_bestfit1, y_bestfit1, label='Fit')
    plt.title('Calibrated ' + title + ' Energy Estimator')
    # plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
    # plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
    # plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    # plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
    # plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)


    # find smallest sigma
    print('c_', title, ' chi2: ', "{:.4f}".format(1-chi2.cdf(chisquared1,dof1)),
          '   sigma: ', "{:00.4f}".format(popt1[2]),
          '   mu: ', "{:00.4f}".format(popt1[1]))

    plt.legend(loc=1)
    if (save): plt.savefig(f'#{count} Calibrated {title}.png', dpi=mydpi)
    plt.show()

    return conv


# graph 1
num_bins_1=60
bin_range_1=(0.25, 0.4)
p_1=[60, 0.31, 0.07, 5]        # myGauss(x, A, mean, width, base)
num_bins_1_c=35
bin_range_1_c=(7, 13)
p_1_c=[80, 10, 2, 0]        # myGauss(x, A, mean, width, base)
plotting_with_calibration(plotting(num_bins_1, bin_range_1, amp1, p_1, 'amp 1'),
                          num_bins_1_c, bin_range_1_c, amp1, p_1_c, 'amp 1', 1)

# graph 2
num_bins_2=35
bin_range_2=(0.16, 0.33)
p_2=(200, 0.23, 0.1, 7)        # myGauss(x, A, mean, width, base)
num_bins_2_c=25
bin_range_2_c=(8, 12)
p_2_c=(50, 10, 2, 5)        # myGauss(x, A, mean, width, base)
plotting_with_calibration(plotting(num_bins_2, bin_range_2, amp2, p_2, 'amp 2'),
                          num_bins_2_c, bin_range_2_c, amp2, p_2_c, 'amp 2', 2)

# graph 3
num_bins_3=30
bin_range_3=(-30, 30)
p_3=(40, 17, 10,0)        # myGauss(x, A, mean, width, base)
num_bins_3_c=50
bin_range_3_c=(-200, 200)
p_3_c=(1, 10, 20, 0)        # myGauss(x, A, mean, width, base)
plotting_with_calibration(plotting(num_bins_3, bin_range_3, area1, p_3, 'area 1'),
                          num_bins_3_c, bin_range_3_c, area1, p_3_c, 'area 1', 3)

# graph 4
num_bins_4=40
bin_range_4=(-200, 200)
p_4=(100,50,150,0)        # myGauss(x, A, mean, width, base)
num_bins_4_c=40
bin_range_4_c=(-100, 100)
p_4_c=(140, 10, 50, 0)        # myGauss(x, A, mean, width, base)
plotting_with_calibration(plotting(num_bins_4, bin_range_4, area2, p_4, 'area 2'),
                          num_bins_4_c, bin_range_4_c, area2, p_4_c, 'area 2', 4)

# graph 5
num_bins_5=40
bin_range_5=(11, 22)
p_5=(60,15,6,0)        # myGauss(x, A, mean, width, base)
num_bins_5_c=40
bin_range_5_c=(2, 15)
p_5_c=(100,10,4,0)        # myGauss(x, A, mean, width, base)
conv_5 = plotting_with_calibration(plotting(num_bins_5, bin_range_5, area3, p_5, 'area 3'),
                          num_bins_5_c, bin_range_5_c, area3, p_5_c, 'area 3', 5)

# graph 6
num_bins_6=40
bin_range_6=(0, 0.4)
p_6=(300,0.25,0.1,0)        # myGauss(x, A, mean, width, base)
num_bins_6_c=45
bin_range_6_c=(1, 19)
p_6_c=(125,10,5,0)         # myGauss(x, A, mean, width, base)
plotting_with_calibration(plotting(num_bins_6, bin_range_6, pulse_fit, p_6, 'pulse_fit'),
                          num_bins_6_c, bin_range_6_c, pulse_fit, p_6_c, 'pulse_fit', 6)

# sigmal: std (around 90%), mu: mean, chi2 (around 1)


'''
Part 3 
'''
stds = []
for i in range(len(noise_data)):
    stds.append(np.std(noise_data['evt_%i'%i]))
std = np.mean(stds)
sig = np.full(len(noise_data['evt_0']), std)
sig = np.where(sig==0, 1, sig)
"""
calculating the std of the noise
"""

for ievt in range(1000):
    current_data = signal_data['evt_%i'%ievt]
    area3[ievt] = np.sum(current_data[1000:1100])
    popt, pcov = curve_fit(fit_pulse, xx, current_data,
                  sigma = sig, absolute_sigma=True)
    pulse_fit[ievt] = popt[0]
pulse_fit*=1000

# num_bins_f = num_bins_6_c
# bin_range_f = bin_range_6_c
# p_f = p_6_c

num_bins_f = 40
bin_range_f = (-0.05, 0.1)
p_f = (120, 0.01, 0.1, 0)

pulse_fit *= conv_5

n1, bin_edges1, _ = plt.hist(pulse_fit, bins=num_bins_f, range=bin_range_f,\
                             color='k', histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use

plt.xlabel('Energy Estimator: Maximum Value (keV)')
plt.ylabel('Events / %2.2f keV'%((bin_range_f[-1]-bin_range_f[0])/num_bins_f));
plt.xlim(bin_range_f)
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])

sig1 = np.sqrt(n1)
sig1=np.where(sig1==0, 1, sig1)
# The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)

popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1,
             sigma = sig1, p0=p_f, absolute_sigma=True)

n1_fit = myGauss(bin_centers1, *popt1)

chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
dof1 = num_bins_f - len(popt1)

x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1)

fontsize=10
plt.plot(x_bestfit1, y_bestfit1, label='Fit')
# plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
# plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
# plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
# plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
# plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
plt.legend(loc=1)
plt.title('Final Signal Reconstruction')
if (save): plt.savefig('#7 Final Signal Reconstruction.png', dpi=mydpi)
plt.show()

print('c_', 'final testing', ' chi2: ', "{:.4f}".format(1 - chi2.cdf(chisquared1, dof1)),
      '   sigma: ', "{:00.4f}".format(popt1[2]),
      '   mu: ', "{:00.4f}".format(popt1[1]))
