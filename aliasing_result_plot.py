# script for plotting the results of the aliasing test

from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
from graphing import plot_specgram, plot_hurst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi, floor
np.set_printoptions(precision=2, suppress=True)

frequency = 10
frequency_2 = 0
fs = 100
t_start = 0
t_end = 2

time = np.linspace(t_start,t_end,(t_end-t_start)*fs)
amplitude = np.sin(time*frequency*2*pi) +np.sin(time*frequency_2*2*pi)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

fig1, [ax1,ax2]=plt.subplots(2,1,sharex=False)
fig1.set_size_inches(5,6)

#create space between the two subplots
fig1.subplots_adjust(top=0.95,right=0.95,left=0.15,bottom=0.09,hspace=0.3)

(Pxx, freqs, line) = ax1.psd(amplitude,Fs=fs,return_line=True,color='#000000')
x,y=line[0].get_data()
# unaliased_amplitude = np.max(y[:2*floor(len(y)/5)])
unaliased_amplitude=y[4*floor(len(y)/5)]
print(unaliased_amplitude)

allWavelets = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']
allWavelets = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5']
goodWavelets = ['db1','db10','sym20']
wavelets_db = ['db'+str(i) for i in range(1,21)]
wavelets_sym = ['sym'+str(i) for i in range(2,21)]
wavelets_coif = ['coif'+str(i) for i in range(1,6)]

alias_vals = ['']*len(allWavelets)

linestyles = ['--',':','--']
colors = ['magenta','magenta','#1ff037']
gwlcount = 0

for i,mother_wavelet in enumerate(allWavelets):
    print(str(mother_wavelet))
    amplitude_denoised = denoise(amplitude,level=1,mother_wavelet=mother_wavelet,threshold_selection='test_aliasing')#_2',intensity=intensity)
    if mother_wavelet in goodWavelets:
        (Pxx, freqs, line) = ax1.psd(amplitude_denoised,Fs=fs,return_line=True,linestyle=linestyles[gwlcount],color=colors[gwlcount])
        gwlcount+=1
    else: 
        (Pxx, freqs, line) = ax0.psd(amplitude_denoised,Fs=fs,return_line=True)

    # get and store data for second plot
    x,y=line[0].get_data()
    aliased_amplitude = np.max(y[4*floor(len(y)/5)-floor(len(y)/20):4*floor(len(y)/5)+floor(len(y)/10)])
    print(str(mother_wavelet)+' '+str(aliased_amplitude), end=" difference: ")
    print(str(aliased_amplitude-unaliased_amplitude))
    alias_vals[i] = aliased_amplitude-unaliased_amplitude


goodWavelets.insert(0,'Original')
legend = ['Original','Daubechies N=2','Daubechies N=20','Symlets N=40']
ax1.legend(legend,loc='upper center',bbox_to_anchor=(0.5,1),frameon=False)
ax1.set_xlim([0,50])
ax1.set_xlabel('Frequency (Hz)')
# turn off the grid
ax1.grid(False)
ax1.set_title('a',loc='left',fontweight='bold')

# second plot
for i in range(len(alias_vals)):
    filters = getFilters(allWavelets[i])
    print(len(filters[0]))
    if allWavelets[i] in wavelets_db:
        color = 'magenta'
    elif allWavelets[i] in wavelets_sym:
        color = '#1ff037'
    elif allWavelets[i] in wavelets_coif:
        color = '#0836bf'
    print(f'plotting {allWavelets[i]} with lenth {len(filters[0])} and value {alias_vals[i]}')
    ax2.plot(len(filters[0]),alias_vals[i],'x',color=color)

symbol = 'x'
colors = ['magenta','#1ff037','#0836bf']
labels = ['Daubechies','Symlets','Coiflets']
handles = []
for i in range(len(colors)):
    #invidible points for legend
    handle,= ax0.plot([0],[0],marker=symbol,color=colors[i],label=labels[i])
    handles.append(handle)

#dashed grey horizontal line at 0:
ax2.axhline(y=0, color='grey', linestyle='--')

ax2.legend(handles=handles,loc='upper right',bbox_to_anchor=(1,1),frameon=False)
ax2.set_xlabel('Filter Length N')
ax2.set_ylabel('Peak difference at 40 Hz (dB/Hz)')
ax2.set_xlim([0,42])
ax2.set_ylim([-20,50])
ax2.set_title('b',loc='left',fontweight='bold')

plt.show()