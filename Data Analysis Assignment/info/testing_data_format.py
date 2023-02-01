import pickle
import matplotlib.pyplot as plt
import numpy as np

save=False
mydpi=300

with open('calibration_p3.pkl','rb') as file:
    data_from_file=pickle.load(file)

xx1 = np.linspace(0, 4095/1e3, 4096)
# "ect_x" where x iterates from 0 to 99, each is a numpy array 4096 long
# each entry of the array is a voltage measuremetn done by 1MHz
plt.plot(xx1, data_from_file['evt_2'], color='k', lw=0.5)
plt.title('evt_2')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
plt.show()

if(save): plt.savefig('calibration_p3_evt_2.png',dpi=mydpi)
plt.show()

##################################################################

with open('noise_p3.pkl','rb') as file:
    data_from_file_2=pickle.load(file)

xx2 = np.linspace(0, 4095/1e3, 4096)
plt.plot(xx2, data_from_file_2['evt_2'], color='k', lw=0.5)
plt.title('Noise')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
plt.show()

if(save): plt.savefig('noise_p3_evt_2.png',dpi=mydpi)
plt.show()

##################################################################

with open('signal_p3.pkl','rb') as file:
    data_from_file_3=pickle.load(file)

xx = np.linspace(0, 4095/1e3, 4096)
# "ect_x" where x iterates from 0 to 99, each is a numpy array 4096 long
# each entry of the array is a voltage measuremetn done by 1MHz
plt.plot(xx, data_from_file_3['evt_2'], color='k', lw=0.5)
plt.title('Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
plt.show()

if(save): plt.savefig('noise_p3_evt_2.png',dpi=mydpi)
plt.show()
