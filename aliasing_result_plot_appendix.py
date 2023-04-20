# script for plotting the results of the aliasing test

from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
from graphing import plot_specgram, plot_hurst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi, floor
np.set_printoptions(precision=2, suppress=True)

allWavelets = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']
allWavelets = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5']
wavelets_db = ['db'+str(i) for i in range(1,21)]
wavelets_sym = ['sym'+str(i) for i in range(2,21)]
wavelets_coif = ['coif'+str(i) for i in range(1,6)]

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)

fig1, axs = plt.subplots(4,4)#,sharex=True,sharey=True)
fig1.set_size_inches(15,12)
#create space between the two subplots
fig1.tight_layout()

fs = 100
t_start = 0
frequency = 10
t_end = 4
t_ends = [1,2,3,4]
frequencies = [5,10,15,20]

for i_f, frequency in enumerate(frequencies):
    for i_t,t_end in enumerate(t_ends):
        ax = axs[i_f,i_t]

        time = np.linspace(t_start,t_end,(t_end-t_start)*fs)
        amplitude = np.sin(time*frequency*2*pi)

        # original vals
        (Pxx, freqs, line) = ax0.psd(amplitude,Fs=fs,return_line=True,color='#000000')
        x,y=line[0].get_data()
        # unaliased_amplitude = np.max(y[:2*floor(len(y)/5)])
        alias_index = floor(len(y)*(1-frequency/(fs/2)))

        print(f'signal frequency {frequency} Hz, alias fraction: {(1-frequency/(fs/2))}, time {t_end} s, alias index {alias_index}')

        unaliased_amplitude=y[alias_index]
        print(unaliased_amplitude)

        alias_vals = ['']*len(allWavelets)

        for i,mother_wavelet in enumerate(allWavelets):
            filters = getFilters(mother_wavelet)
            print(f'plotting {mother_wavelet} with lenth {len(filters[0])}')
            amplitude_denoised = denoise(amplitude,level=1,mother_wavelet=mother_wavelet,threshold_selection='test_aliasing')#_2',intensity=intensity)

            (Pxx, freqs, line) = ax0.psd(amplitude_denoised,Fs=fs,return_line=True)

            # get data for alias plot
            x,y=line[0].get_data()
            aliased_amplitude = np.max(y[ alias_index -floor(len(y)/20) : alias_index +floor(len(y)/20) ])

            print(str(mother_wavelet)+' '+str(aliased_amplitude), end=" difference: ")
            print(str(aliased_amplitude-unaliased_amplitude))

            alias_vals[i] = aliased_amplitude-unaliased_amplitude

            if mother_wavelet in wavelets_db:
                color = 'magenta'
            elif mother_wavelet in wavelets_sym:
                color = '#1ff037'
            elif mother_wavelet in wavelets_coif:
                color = '#0836bf'

            ax.plot(len(filters[0]),alias_vals[i],'x',color=color)

        symbol = 'x'
        colors = ['magenta','#1ff037','#0836bf']
        labels = ['Daubechies','Symlets','Coiflets']
        handles = []
        for i in range(len(colors)):
            #invidible points for legend
            handle,= ax0.plot([0],[0],marker=symbol,color=colors[i],label=labels[i])
            handles.append(handle)

        #dashed grey horizontal line at 0:
        ax.axhline(y=0, color='grey', linestyle='--')

        # ax.legend(handles=handles,loc='upper right',bbox_to_anchor=(1,1),frameon=False)
        
        ax.set_xlim([0,42])
        # ax.set_ylim([-20,50])
        ax.set_title(f'f={frequency} Hz, t={t_end} s',loc='left',fontweight='bold')

# add legend to the right of the plot
fig1.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,1),frameon=False,ncol=3,fontsize=14)

#set global axes labels
fig1.text(0.5, 0.04, 'Wavelet Filter length N', ha='center', va='center',fontsize=14)
fig1.text(0.04, 0.5, 'Amplitude difference at aliased frequency (dB/Hz)', ha='center', va='center', rotation='vertical', fontsize=14)
fig1.subplots_adjust(left=0.076, bottom=0.076, right=0.95, top=0.93,hspace=0.28)

plt.show()