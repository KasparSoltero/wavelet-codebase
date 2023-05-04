# script for plotting the results of the aliasing test

from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
from graphing import plot_specgram, plot_hurst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from math import pi, floor
np.set_printoptions(precision=2, suppress=True)

time_mwavelet_data = 'wavelet-codebase\wavelet_results_' + 'time' + '.csv'
# open data out for reading
with open(time_mwavelet_data, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    #delete first row
    data.pop(0)
    #convert to float
    print(data)
    data = [[float(x) for x in row[1:]] for row in data]
    print(data)
filter_lengths = [row[0] for row in data]
times = [row[1] for row in data]

print(filter_lengths)
print(times)

fig0 = plt.figure()
ax = fig0.add_subplot(111)
fig0.set_size_inches(5,3)

fig0.subplots_adjust(top=0.967,right=0.952,left=0.133,bottom=0.167)

allWavelets = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5']
wavelets_db = ['db'+str(i) for i in range(1,21)]
wavelets_sym = ['sym'+str(i) for i in range(2,21)]
wavelets_coif = ['coif'+str(i) for i in range(1,6)]

linestyles = ['--',':','-.']
colors = ['magenta','#1ff037','#0836bf']
labels = ['Daubechies','Symlets','Coiflets']

for i in range(len(allWavelets)):
    if allWavelets[i] in wavelets_db:
        color = colors[0]
        label = labels[0]
    elif allWavelets[i] in wavelets_sym:
        color = colors[1]
        label = labels[1]
    elif allWavelets[i] in wavelets_coif:
        color = colors[2]
        label = labels[2]
    ax.plot(filter_lengths[i],times[i],'x',color=color,label=label)


ax.set_xlabel('Filter length $L$')
ax.set_ylabel('Mean computation time (ms)')
handles = []
for i in range(len(colors)):
    #invidible points for legend
    handle,= ax.plot([0],[0],marker='x',color=colors[i],label=labels[i])
    handles.append(handle)

ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1),frameon=False)

plt.show()