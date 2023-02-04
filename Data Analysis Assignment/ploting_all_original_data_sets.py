import pickle
import matplotlib.pyplot as plt
import numpy as np

save=True
mydpi=300

with open('calibration_p3.pkl','rb') as file:
    data_from_file=pickle.load(file)

xx1 = np.linspace(0, 4095/1e3, 4096)
# "ect_x" where x iterates from 0 to 99, each is a numpy array 4096 long
# each entry of the array is a voltage measuremetn done by 1MHz
for i in data_from_file:
    plt.plot(xx1, data_from_file[i], lw=0.5)
plt.title('Calibration All Data')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#10.1 - Calibration_p3_All_Data.png',dpi=mydpi)
plt.show()

plt.plot(xx1, data_from_file['evt_2'], color='k', lw=0.5)
plt.title('Calibration Data[evt_2]')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#10.2 - Calibration_p3_[evt_2].png',dpi=mydpi)
plt.show()

##################################################################

with open('noise_p3.pkl','rb') as file:
    data_from_file_2=pickle.load(file)

xx2 = np.linspace(0, 4095/1e3, 4096)
for i in data_from_file:
    plt.plot(xx1, data_from_file[i], lw=0.5)
plt.title('Noise All Data')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#11.1 - Noise_p3_All_Data.png',dpi=mydpi)
plt.show()

plt.plot(xx2, data_from_file_2['evt_2'], color='k', lw=0.5)
plt.title('Noise[evt_2]')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#11.2 - Noise_p3_[evt_2].png',dpi=mydpi)
plt.show()

##################################################################

with open('signal_p3.pkl','rb') as file:
    data_from_file_3=pickle.load(file)

xx = np.linspace(0, 4095/1e3, 4096)
# "ect_x" where x iterates from 0 to 99, each is a numpy array 4096 long
# each entry of the array is a voltage measuremetn done by 1MHz
for i in data_from_file:
    plt.plot(xx1, data_from_file[i], lw=0.5)
plt.title('Signal All Data')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#12.1 - Signal_p3_All_Data.png',dpi=mydpi)
plt.show()

plt.plot(xx, data_from_file_3['evt_2'], color='k', lw=0.5)
plt.title('Signal[evt_2]')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
if(save): plt.savefig('#12.2 - Signal_p3_[evt_2].png',dpi=mydpi)
plt.show()
