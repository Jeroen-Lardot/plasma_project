import numpy as np
import matplotlib.pyplot as plt

info_all = np.load('BIC_info.npy', allow_pickle=True)[0]

E_therm = np.load('Etherm_biccomponents.npy',allow_pickle=True)[0]

#for some reason, I can't just plot info_all, something wrong with the type. I had to do it like this
info_plot = []
for row in info_all:
    info_plot.append(list(row))

cf=plt.imshow(np.transpose(info_plot), origin = 'upper', cmap='jet', aspect='auto')
plt.xlabel('time')
plt.ylabel('n of clusters')
plt.colorbar()
plt.show()

plt.plot(E_therm)
plt.show()