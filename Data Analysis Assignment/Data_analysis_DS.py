import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.stats as stats
with open("calibration_p3.pkl",'rb') as file:
    data_from_file=pickle.load(file)

with open("noise_p3.pkl",'rb') as file:
    noise_from_file=pickle.load(file)

with open("signal_p3.pkl",'rb') as file:
    signal_from_file=pickle.load(file)


def estimator1(data_from_file):
    """
    :param data_from_file: data from file
    :return: returns amplitude of the data file as calculated by max-min
    """
    amplitudes = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        peak = max(this_data)
        trough = min(this_data)
        amplitudes.append(1000*(peak-trough))
        noise_data = noise_from_file[key]
        uncert = np.std(noise_data)
        uncert_in_value.append(uncert)
    return amplitudes, uncert_in_value


def estimator2(data_from_file):
    """
    :param data_from_file: data from file
    :return: returns amplitude of the data file as calculated by max-baseline
    """
    amplitudes = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        peak = max(this_data)
        average = np.average(this_data[:1000])
        amplitudes.append(1000*(peak-average))
        noise_data = noise_from_file[key]
        uncert = np.std(noise_data)
        uncert_in_value.append(uncert)
    return amplitudes, uncert_in_value


def estimator3(data_from_file):
    """
    :param data_from_file: data from file
    :return: returns integral of whole trace
    """
    integral = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        add = sum(this_data)
        integral.append(1000*add)
        noise_data = noise_from_file[key]
        uncert = np.std(noise_data)
        uncert_in_value.append(uncert)
    return integral, uncert_in_value


def estimator4(data_from_file):
    """
    :param data_from_file: data from file
    :return: returns integral of whole trace subtract the base line
    """
    integral = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        add = sum(this_data)
        baseline = np.average(this_data[:1000])
        integral.append(1000*(add-baseline))
        noise_data = noise_from_file[key]
        uncert = np.std(noise_data)
        uncert_in_value.append(uncert)
    return integral, uncert_in_value


def estimator5(data_from_file):
    """
    :param data_from_file: data from file
    :return: returns integral of whole trace after limiting the range
    """
    integral = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        add = sum(this_data[1000:1100])
        integral.append(1000*add)
        noise_data = noise_from_file[key]
        uncert = np.std(noise_data)
        uncert_in_value.append(uncert)
    return integral, uncert_in_value


def model_func(time, A):
    """
    :param A: Without this, the amplitude of this function is 1
    :return: A model function that depicts the pulse
    """
    C = ((0.02-0.08)/0.08)*(0.08/0.02)**(-0.02/(0.08-0.02))
    return A*C*(np.e**(-(time-1)/0.02)-np.e**(-(time-1)/0.08))


def estimator6(data_from_file, noise_from_file):
    """
    :param data_from_file: data from file
    :return: does a chi sq fit on the pulse data and returns the amplitude
    """
    amplitude = []
    uncert_in_value = []
    for j in range(1000):
        key = 'evt_' + str(j)
        this_data = data_from_file[key]
        noise_data = noise_from_file[key]
        xx = np.linspace(0, 4095/1e4, 4096)
        uncert = np.std(noise_data)
        uncert_y = np.ones(len(xx))*uncert
        opt_vals, opt_cov = scipy.optimize.curve_fit(model_func, xx[1000:], data_from_file[key][1000:], p0=None, sigma=uncert_y[1000:], absolute_sigma=True)
        A = opt_vals
        peak = np.max(model_func(xx[1000:], A))
        trough = np.min(model_func(xx[1000:], A))
        amplitude.append(1000*(peak-trough))
        uncert_in_value.append(uncert)
    return amplitude, uncert_in_value


def gaussian_fit(x, amplitude, mean, std):
    return amplitude*np.e**((-(x-mean)**2)/(2*std**2))


def chi_sq(y, x, uncert):
    """ y: modelled data f(x),
        x: experimental data
        uncert: uncertainty in y (biggest uncertainty)
    """
    numerator = (y-x)**2
    # denominator = uncert**2
    terms = []
    for i in range(len(uncert)):
        if uncert[i] == 0:
            uncert[i] = 1
        terms.append(numerator[i]/uncert[i]**2)

    # chi_sq_terms = np.divide(numerator, denominator)
    chi_sq = np.sum(terms)
    return chi_sq

# """Calibration 1: doing 2a for max - min"""
# values0 = estimator1(data_from_file)[0]  # this is giving me the amplitudes of my data which I will use to estimate energy
# plt.hist(values0, bins =40, label="Data", edgecolor='black')  # plotting the histogram
# plt.xlabel('Amplitude (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Amplitude for max - min")
# xx = np.linspace(min(values0), max(values0), 40)  # this will be my x-values on the graph. starts from the min of amplitude to max of amplitude and has 40 data points in between them
# data = np.histogram(values0, bins=40)  #data[0] gives me the heights of my bins
# # print(data[0])
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #curve fitting my gaussian to the histogram
# A,m,s = opt_vals  # A is the amplitude, m is the mean, s is the std of gaussian
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (.1,160))
# plt.annotate(f"σ={s:.3f}", (.1,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (.1,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (.1,130))
# plt.legend()
# plt.show()
#
# #------------------Energy estimator------------------
# caliberator = 10/m
# print(caliberator)
# values00 = np.multiply(estimator1(data_from_file)[0],caliberator)
# plt.hist(values00, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.45 keV')
# plt.title("Energy calibration 1")
# xx = np.linspace(min(values00), max(values00), 40)
# data = np.histogram(values00, bins=40)
# print(data)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (4,160))
# plt.annotate(f"σ={s:.3f}", (4,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (4,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (4,130))
# plt.legend()
# plt.show()
#
#
# """Calibration 2: doing 2a for max - baseline"""
# values1 = estimator2(data_from_file)[0]
# plt.hist(values1, bins =40, label="Data", edgecolor='black')
#
# plt.xlabel('Amplitude (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Amplitude for max - baseline")
# xx = np.linspace(min(values1), max(values1), 40)
# data = np.histogram(values1, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None)
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))
# dof = (len(data[0])-len(opt_vals))
# red_chi_sq = chi_sq1/dof
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (.1,160))
# plt.annotate(f"σ={s:.3f}", (.1,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (.1,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (.1,130))
# plt.legend()
# plt.show()
#
# #------------------Energy estimator------------------
# caliberator1 = 10/m
# print(caliberator1)
# values01 = np.multiply(estimator2(data_from_file)[0],caliberator1)
# plt.hist(values01, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.5 keV')
# plt.title("Energy calibration 2")
# xx = np.linspace(min(values01), max(values01), 40)
# data = np.histogram(values01, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (4,160))
# plt.annotate(f"σ={s:.3f}", (4,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (4,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (4,130))
# plt.legend()
# plt.show()
# #
#
# """Calibration 3: doing 2a for Integral 1"""
# values2 = estimator3(data_from_file)[0]
# plt.hist(values2, bins =40, label="data", edgecolor='black')
# plt.xlabel('Integral (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Integral for whole trace")
# xx = np.linspace(min(values2), max(values2), 40)
# data = np.histogram(values2, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))
# dof = (len(data[0])-len(opt_vals))
# red_chi_sq = chi_sq1/dof
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (-140,80))
# plt.annotate(f"σ={s:.3f}", (-140,70))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (-140,60))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (-140,50))
# plt.legend()
# plt.show()
# #------------------Energy estimator------------------
# caliberator = 10/m
# print(caliberator)
# values01 = np.multiply(estimator3(data_from_file)[0],caliberator)
# plt.hist(values01, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.45 keV')
# plt.title("Energy calibration 3")
# xx = np.linspace(min(values01), max(values01), 40)
# data = np.histogram(values01, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (-40,80))
# plt.annotate(f"σ={s:.3f}", (-40,70))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (-40,60))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (-40,50))
# plt.legend()
# plt.show()
#
#
# """Calibration 4: doing 2a for integral 2"""
# values3 = estimator4(data_from_file)[0]
# plt.hist(values3, bins =40, label="Data", edgecolor='black')
# plt.xlabel('Integral (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Integral for whole trace - baseline")
# xx = np.linspace(min(values3), max(values3), 40)
# data = np.histogram(values3, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))
# dof = (len(data[0])-len(opt_vals))
# red_chi_sq = chi_sq1/dof
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (-140,80))
# plt.annotate(f"σ={s:.3f}", (-140,70))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (-140,60))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (-140,50))
# plt.legend()
# plt.show()
# #------------------Energy estimator------------------
# caliberator = 10/m
# print(caliberator)
# values01 = np.multiply(estimator4(data_from_file)[0],caliberator)
# plt.hist(values01, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.45 keV')
# plt.title("Energy calibration 4")
# xx = np.linspace(min(values01), max(values01), 40)
# data = np.histogram(values01, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (-40,80))
# plt.annotate(f"σ={s:.3f}", (-40,70))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (-40,60))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (-40,50))
# plt.legend()
# plt.show()
#
#
# """Calibration 5: doing 2a for integral 3"""
# values4 = estimator5(data_from_file)[0]
# plt.hist(values4, bins =40, label="Data", edgecolor='black')
# plt.xlabel('Integral (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Integral for selected part of trace")
# xx = np.linspace(min(values4), max(values4), 40)
# data = np.histogram(values4, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))
# dof = (len(data[0])-len(opt_vals))
# red_chi_sq = chi_sq1/dof
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (.1,160))
# plt.annotate(f"σ={s:.3f}", (.1,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (.1,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (.1,130))
# plt.legend()
# plt.show()
#
# #------------------Energy estimator------------------
# caliberator = 10/m
# print(caliberator)
# values01 = np.multiply(estimator5(data_from_file)[0],caliberator)
# plt.hist(values01, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.45 keV')
# plt.title("Energy calibration 5")
# xx = np.linspace(min(values01), max(values01), 40)
# data = np.histogram(values01, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (1,160))
# plt.annotate(f"σ={s:.3f}", (1,150))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (1,140))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (1,130))
# plt.legend()
# plt.show()
#
# """Calibration 6: doing 2a for fit amplitude"""
# values5 = estimator6(data_from_file, noise_from_file)[0]
# plt.hist(values5, bins =40, label="Data", edgecolor='black')
# plt.xlabel('Amplitude (mV)')
# plt.ylabel('Events / 0.01 mV')
# plt.title("Amplitude for curve fit max - min")
# xx = np.linspace(min(values5), max(values5), 40)
# data = np.histogram(values5, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))
# dof = (len(data[0])-len(opt_vals))
# red_chi_sq = chi_sq1/dof
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (.01,100))
# plt.annotate(f"σ={s:.3f}", (.01,90))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (.01,80))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (.01,70))
# plt.legend()
# plt.show()
#
# #------------------Energy estimator------------------
# caliberator = 10/m
# print(caliberator)
# values01 = np.multiply(estimator6(data_from_file,noise_from_file)[0],caliberator)
# plt.hist(values01, bins =40, edgecolor='black', label='Data')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events / 0.45 keV')
# plt.title("Energy calibration 6")
# xx = np.linspace(min(values01), max(values01), 40)
# data = np.histogram(values01, bins=40)
# opt_vals, opt_cov = scipy.optimize.curve_fit(gaussian_fit, xx, data[0], p0=None, absolute_sigma=True)  #sigma=np.ones(len(xx)),
# A,m,s = opt_vals
# print(A,m,s)
# chi_sq1= chi_sq(gaussian_fit(xx, A, m, s), data[0], np.sqrt(data[0]))  # Gives me the chi sq of my gaussian fit
# dof = (len(data[0])-len(opt_vals))  # degreed of freedom
# red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
# plt.plot(xx, gaussian_fit(xx, A, m, s), label='Fit')  # plots the gaussian
# plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
# plt.annotate(f"μ={m:.3f}", (2,110))
# plt.annotate(f"σ={s:.3f}", (2,100))
# plt.annotate(f"χ2={red_chi_sq:.3f}", (2,90))
# plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (2,80))
# plt.legend()
# plt.show()
#

def new_func(x, A, tao):
    """
    sum of an exponential decay and gaussian
    :return:
    """
    z = A*(np.e**(-x*tao))
    k = 30*np.e**((-(x-0.15)**2)/(2*0.014**2))
    return z+k+3


#_____________Part 3___________
values1 = estimator2(signal_from_file)[0]
plt.hist(values1, bins =40, label="Signal Data", edgecolor='black')

plt.xlabel('Amplitude (mV)')
plt.ylabel('Events / 0.01 mV')
plt.title("Signal data + background")
xx = np.linspace(min(values1), max(values1), 40)
data = np.histogram(values1, bins=40)
print(data[0]) #[701:869]
print(len(xx))
uncert = np.sqrt(data[0])
opt_vals, opt_cov = scipy.optimize.curve_fit(new_func, xx[1:], data[0][1:], p0=None)
A,tao = opt_vals
sigma = np.sqrt(np.diag(opt_cov))
print(A,tao)
print(sigma)
chi_sq1= chi_sq(new_func(xx[1:], A, tao), data[0][1:], np.sqrt(data[0])[1:])
dof = (len(data[0])-len(opt_vals))
red_chi_sq = chi_sq1/dof
plt.plot(xx[1:], new_func(xx[1:], A, tao), label='Fit')
plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
plt.annotate(f"red χ2={red_chi_sq:.3f}", (.1,140))
plt.annotate(f"tao={tao:.3f} ± {sigma[1]:.3f}", (.1,150))
plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (.1,130))
plt.legend()
plt.show()

def new_func(x, A, tao):
    """
    sum of an exponential decay and gaussian
    :return:
    """
    z = A*(np.e**(-x*tao))
    k = 31*np.e**((-(x-6.43)**2)/(2*0.658**2))
    return z+k+3
#------------------Energy estimator------------------
values1 = np.multiply(estimator2(signal_from_file)[0],41.84)
plt.hist(values1, bins =40, edgecolor='black', label='Data')
plt.xlabel('Energy (keV)')
plt.ylabel('Events / 0.5 keV')
plt.title("Signal energy")
xx = np.linspace(min(values1), max(values1), 40)
data = np.histogram(values1, bins=40)
opt_vals, opt_cov = scipy.optimize.curve_fit(new_func, xx[1:], data[0][1:], p0=None)  #sigma=np.ones(len(xx)),
A,tao = opt_vals
print(A,tao)
sigma = np.sqrt(np.diag(opt_cov))
print(sigma)
chi_sq1= chi_sq(new_func(xx[1:], A, tao), data[0][1:], np.sqrt(data[0])[1:])  # Gives me the chi sq of my gaussian fit
dof = (len(data[0])-len(opt_vals))  # degreed of freedom
red_chi_sq = chi_sq1/dof  # reduced chi square of my fit
plt.plot(xx[1:], new_func(xx[1:], A, tao), label='Fit')  # plots the gaussian
plt.errorbar(xx, data[0], yerr=np.sqrt(data[0]), fmt='.')
plt.annotate(f"tao={tao:.3f} ± {sigma[1]:.3f}", (4,150))
plt.annotate(f"red χ2={red_chi_sq:.3f}", (4,140))
plt.annotate(f"χ2prob={1 - stats.chi2.cdf(chi_sq1, dof):.3f}", (4,130))
plt.legend()
plt.show()


#
# xx = np.linspace(0, 4095/1e3, 4096)
# noise_data = noise_from_file['evt_9']
# uncert = np.std(noise_data)
# uncert_y = np.ones(len(xx))*uncert
# opt_vals, opt_cov = scipy.optimize.curve_fit(model_func, xx[1000:], data_from_file['evt_9'][1000:], p0=None, sigma=uncert_y[1000:], absolute_sigma=True)
# A = opt_vals
# print(A, uncert_y)
# plt.plot(xx[1000:], model_func(xx[1000:],A), label="curvefit")
# plt.plot(xx, data_from_file['evt_9'], label="data")
# plt.legend()
# peak = np.max(model_func(xx[1000:], A))
# trough = np.min(model_func(xx[1000:], A))
# print(peak-trough)
# #
# # for i in range(10):
# #     evt = 'evt_' + str(i)
# #     plt.plot(xx, data_from_file[evt], lw=0.5)
#
# plt.xlabel('Time (ms)')
# plt.ylabel('Readout Voltage (V)')
# plt.title("10 readings from the calibration data set")
# # plt.ylim(0,0.0001)
# plt.show()


'''Task 3'''

