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


def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
# This is my fitting function, a Guassian with a uniform background.

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
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
with open("signal_p3.pkl","rb") as file:
    signal=pickle.load(file)
with open("noise_p3.pkl","rb") as file:
    noise=pickle.load(file)

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

amp1=np.zeros(1000)
amp2=np.zeros(1000)
area1=np.zeros(1000)
area2=np.zeros(1000)
area3=np.zeros(1000)
pulse_fit=np.zeros(1000)
amps = [amp1, amp2, area1, area2, area3, pulse_fit]

stds = []
for i in range(len(noise)):
    stds.append(np.std(noise['evt_%i'%i]))
std = np.mean(stds)
sig = np.full(len(noise['evt_0']), std)
sig = np.where(sig==0, 1, sig) 

xx=np.linspace(0, 4095, 4096)
for ievt in range(1000):
    current_data = calibration_data['evt_%i'%ievt]
    amp1_calculation = np.max(current_data)-np.min(current_data)
    amp1[ievt] = amp1_calculation
    
    base = np.mean(current_data[:1000])
    amp2[ievt] = np.max(current_data)-base
    area1[ievt] = sum(current_data)
    area2[ievt] = sum(current_data)-base
    area3[ievt] = sum(current_data[1000:1300])
    
    popt, pcov = curve_fit(fit_pulse, xx, current_data, 
                  sigma = sig, absolute_sigma=True)
    pulse_fit[ievt] = popt[0]
    
for amp in amps:
    amp*=1000
# convert from V to mV   


num_bins1=40 
bin_range1=(0.2,0.4) 
p=(100,0.01,0.4,5) # (a,b,c,d)
def main(num_bins1, bin_range1, amps, p):

    """
    These two values were picked by trial and error. You'll 
    likely want different values for each estimator.
    """
    
    n1, bin_edges1, _ = plt.hist(amps, bins=num_bins1, range=bin_range1,\
                                 color='k', histtype='step', label='Data')
    # This plots the histogram AND saves the counts and bin_edges for later use
    
    plt.xlabel('Energy Estimator: Maximum Value (mV)')
    plt.ylabel('Events / %2.2f mV'%((bin_range1[-1]-bin_range1[0])/num_bins1));
    plt.xlim(bin_range1)  
    # If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
    
    bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
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
    sig1=np.where(sig1==0, 1, sig1) 
    # The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.
    
    plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
    # This adds errorbars to the histograms, where each uncertainty is sqrt(y)
    
    popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1, 
                 sigma = sig1, p0=p, absolute_sigma=True)
    
    
    n1_fit = myGauss(bin_centers1, *popt1)
    """
    n1_fit is our best fit line using our data points.
    Note that if you have few enough bins, this best fit
    line will have visible bends which look bad, so you
    should not plot n1_fit directly. See below.
    """
    
    chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
    dof1 = num_bins1 - len(popt1)
    # Number of degrees of freedom is the number of data points less the number of fitted parameters
    
    x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y_bestfit1 = myGauss(x_bestfit1, *popt1) 
    # Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 10 data points!
    
    print('mu = %3.4f'%(popt1[1]),"\r\n", 
          'sigma = %3.4f'%(popt1[2]),"\r\n",
          '%3.4f/%i'%(chisquared1,dof1),"\r\n",
          'chi^2 prob.= %3.4f'%(1-chi2.cdf(chisquared1,dof1)))
    fontsize=10
    plt.plot(x_bestfit1, y_bestfit1, label='Fit')
    # plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
    # plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
    # plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    # plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
    # plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
    plt.legend(loc=1)
    plt.show()
    # plt.figure()
    
    
    conv = 10/popt1[1]
    
    return popt1
    
    
    #-------------- Converted to 10KeV ------------------
    

def MAIN(popt1, num_bins1, bin_range1, amps, a,b,c,d):
    
    # energy_amp1 = amp1 * conversion_factor1
    conv = 10/popt1[1]
    amps *= conv
    
    
    n1, bin_edges1, _ = plt.hist(amps, bins=num_bins1, range=bin_range1,\
                                 color='k', histtype='step', label='Data')
    # This plots the histogram AND saves the counts and bin_edges for later use
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Events / %2.2f keV'%((bin_range1[-1]-bin_range1[0])/num_bins1));
    plt.xlim(bin_range1)
    # If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
    
    bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
    
    
    
    sig1 = np.sqrt(n1)
    sig1=np.where(sig1==0, 1, sig1) 
    # The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.
    
    plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
    # This adds errorbars to the histograms, where each uncertainty is sqrt(y)
    
    
    popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1, 
                 sigma = sig1, p0=(a,conv*b,conv*c,d), absolute_sigma=True)
    
    
    n1_fit = myGauss(bin_centers1, *popt1)
    
    
    chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
    dof1 = num_bins1 - len(popt1)
    
    
    x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y_bestfit1 = myGauss(x_bestfit1, *popt1) 
    
    
    print('mu = %3.4f'%(popt1[1]),"\r\n", 
          'sigma = %3.4f'%(popt1[2]),"\r\n",
          '%3.4f/%i'%(chisquared1,dof1),"\r\n",
          'chi^2 prob.= %3.4f'%(1-chi2.cdf(chisquared1,dof1)))
    fontsize=10
    plt.plot(x_bestfit1, y_bestfit1, label='Fit')
    # plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
    # plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
    # plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    # plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
    # plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
    plt.legend(loc=1)
    plt.show()
    
    Eest = [conv , popt1[2], (1-chi2.cdf(chisquared1,dof1))]
    
    return conv , popt1[2], (1-chi2.cdf(chisquared1,dof1)), Eest


num_bins1=40 

bin_range1 = (5,14) 
p=(100,0.1,0.4,5)
(a,b,c,d) = (200,0.01,0.4,5)
MAIN((main(num_bins1, (0.2,0.4), amps[0],p))[:2], num_bins1, bin_range1, amps[0], a,b,c,d)

# p=(200,0.25,0.05,5)
# bin_range1 = (2,15) 
# (a,b,c,d) = (200,0.25,0.05,5)
# MAIN((main(num_bins1, (0.05,0.45), amps[1],p))[:2], num_bins1, bin_range1, amps[1], a,b,c,d)


# bin_range1 = (-50,50) 
# p=(100,10,100,5)
# (a,b,c,d) = (100,10,100,5)
# MAIN((main(num_bins1, (-150,150), amps[2],p))[:2], num_bins1, bin_range1, amps[2], a,b,c,d)


# bin_range1 = (-50,50) 
# p=(100,10,100,5)
# (a,b,c,d) = (200,10,100,5)
# MAIN((main(num_bins1, (-150,150), amps[3],p))[:2], num_bins1, bin_range1, amps[3], a,b,c,d)


# bin_range1 = (0,20) 
# p=(100,10,100,5)
# (a,b,c,d) = p
# MAIN((main(num_bins1, (-10,60), amps[4],p))[:2], num_bins1, bin_range1, amps[4], a,b,c,d)


# bin_range1 = (5,14) 
# p=(100,0.1,0.4,5)
# (a,b,c,d) = p
# MAIN((main(num_bins1, (0,0.4), amps[5],p))[:2], num_bins1, bin_range1, amps[5], a,b,c,d)




# --------------- Part 3 -------------


stds = []
for i in range(len(noise)):
    stds.append(np.std(noise['evt_%i'%i]))
std = np.mean(stds)
sig = np.full(len(noise['evt_0']), std)
sig = np.where(sig==0, 1, sig) 


for ievt in range(1000):
    current_data = signal['evt_%i'%ievt]
    popt, pcov = curve_fit(fit_pulse, xx, current_data, 
                  sigma = sig, absolute_sigma=True)
    pulse_fit[ievt] = popt[0]
pulse_fit*=1000
amp = pulse_fit


bin_range1 = (-5,12.5) 
(a, b,c,d)= (100,1,4,5)
p=(a,b,c,d)


conv = (main(num_bins1, (2,20), amps[0],p))[0]
amp *= conv


n1, bin_edges1, _ = plt.hist(amp, bins=num_bins1, range=bin_range1,\
                             color='k', histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use

plt.xlabel('Energy Estimator: Maximum Value (mV)')
plt.ylabel('Events / %2.2f mV'%((bin_range1[-1]-bin_range1[0])/num_bins1));
plt.xlim(bin_range1)
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])



sig1 = np.sqrt(n1)
sig1=np.where(sig1==0, 1, sig1) 
# The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)


popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1, 
             sigma = sig1, p0=p, absolute_sigma=True)

n1_fit = myGauss(bin_centers1, *popt1)


chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
dof1 = num_bins1 - len(popt1)


x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1) 


print('mu = %3.4f'%(popt1[1]),"\r\n", 
      'sigma = %3.4f'%(popt1[2]),"\r\n",
      '%3.4f/%i'%(chisquared1,dof1),"\r\n",
      'chi^2 prob.= %3.4f'%(1-chi2.cdf(chisquared1,dof1)))
fontsize=10
plt.plot(x_bestfit1, y_bestfit1, label='Fit')
# plt.text(0.01, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
# plt.text(0.01, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
# plt.text(0.01, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
# plt.text(0.01, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
# plt.text(0.01, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()





















