from scipy.special import j0
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit


# /Users/alexzeng/Documents/Academic/PHY324 Github/Thermal Diffusivity Lab/
trial = ['Thermal Diffusivity - Trial #1_ 15s interval.csv',
         'Thermal Diffusivity - Trial #2_ 30s interval.csv',
         'Thermal Diffusivity - Trial #3_ 45s interval.csv',
         'Thermal Diffusivity - Trial #4_ 60s interval.csv',
         'Thermal Diffusivity - Trial #6_ 15s.csv',
         ]

datas = []
ts = []
for i in range(len(trial)):
    with open(trial[i], newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data = data[1:]
    main = []
    t = []
    for i in range(len(data)):
        main.append(float(data[i][2]))
        t.append(float(data[i][1]))
    datas.append(main)
    ts.append(t)

# for i in range(len(datas)):
#     plt.plot(ts[i], datas[i])
print('t[2]: ', ts[2])
env_t = np.linspace(0, 800, 800000, endpoint = True)
plt.plot(env_t, signal.square(2 * np.pi * 5 * env_t))


plt.plot(ts[2],datas[2])
plt.ylim(20,65)
plt.show()

# popt2, pcov2 = curve_fit(j0, ts[2], datas[2])
# plt.plot(ts[2], j0())

