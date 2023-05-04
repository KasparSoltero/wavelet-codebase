import os
from scipy.io import wavfile
from graphing import *
from math import pi, floor
from time import time
from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
from graphing import *
import numpy as np
from pydub import AudioSegment
import scipy

# Set paths
sounds_path = './sounds'
possum_path = os.path.join(sounds_path,'Possum T. vulpecula 2s')
cat_path = os.path.join(sounds_path,'Cat F. Catus 2s')
bird_path = os.path.join(sounds_path, 'bird_2s')
anuran_path = os.path.join(sounds_path, 'anuran_2s')
munually_cleaned_path = os.path.join(sounds_path,'Manually Cleaned')
artificial_noise_added_path = os.path.join(sounds_path,'Artificial Noise')
data_out_path = './wavelet_results.xlsx'

# define which animal sounds to use
sound_file_path = possum_path

### Opens selected files in order 00 - 01
for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
    ### which files to see:
    # if not filename in ['.DS_Store', 'times.txt']:
    if filename=='34.wav':

        ### load file & data
        pathname = os.path.join(sound_file_path,filename)
        print(pathname)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2] #make sure all files are 2 seconds exactly
        # print('{:<20},{:<20}'.format(filename,len(signal)))
        signal=signal.astype(object) #allows numpy to be more precise?
        recording = signal

# for technique in ['multi_res','full_res','scaled_res']:
technique = 'scaled_res'
level=5
windows = 5

levels,level_depths = getScaledDecompositionLevels(signal,max_level=level)    

fig,axs=plt.subplots(1,1)
fig.set_size_inches(5,3)

ax=axs
vmax = 40
vmin = -30
cmap='Greys'
    
pxx,  freq, t, cax = ax.specgram(signal, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
fig.colorbar(cax,ax=ax).set_label('Intensity [dB]')

linestyle='-'
edgecolor='magenta'
lw = 1 #line width
windows_count = windows

[lpf_R,lpf_D,hpf_R,hpf_D] = getFilters('haar')
coeffs,lengthsArray = scaledres(signal,levels,lpf_D,hpf_D)
min_bandwidth = int((fs/2)/(2**level))
print(f'min bw {min_bandwidth}')
H_arrays = []
horizontal_indices_seconds = []
for frequency_band in coeffs:
    thres_array, H_indices, H_array = getHurstThresholds(frequency_band,max_windows=windows_count)
    H_arrays.append(H_array)
    horizontal_indices_seconds.append(2*np.divide(H_indices,len(frequency_band)))

frequency_counter = 0
freq_to_draw = 0
i=0
for val in level_depths:
    if frequency_counter>= freq_to_draw:
        bw = min_bandwidth*2**(level-val)
        # ax.add_patch(Rectangle((0,freq_to_draw), 2, bw, edgecolor=edgecolor, linestyle=linestyle, fill=False, lw=lw))
        ax.plot([0,2],[freq_to_draw,freq_to_draw],color=edgecolor,linestyle=linestyle,lw=lw)
        freq_to_draw += bw
        i+=1
    frequency_counter += min_bandwidth
    
ax.set_xlim([0,2])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (kHz)')
# make x ticks 0.5,1,1.5 etc
ax.set_xticks(np.arange(0,2.1,0.5))
scale = 1e3                     # KHz
ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
ax.yaxis.set_major_formatter(ticks)
ax.set_title('PSD Scaled Frequency Resolution')

plt.subplots_adjust(left=0.136,bottom=0.162,right=0.97,top=0.924)
############################################################################################################

fig2,axs=plt.subplots(1,2,gridspec_kw={'width_ratios': [25,1]})
fig2.set_size_inches(5,3)
ax=axs[0]
pxx,  freq, t, cax = ax.specgram(signal, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
# fig.colorbar(cax).set_label('Intensity [dB]')

edgecolor='#fff'
lw = 0 #line width
windows_count = windows

[lpf_R,lpf_D,hpf_R,hpf_D] = getFilters('haar')
coeffs,lengthsArray = scaledres(signal,levels,lpf_D,hpf_D)

globalMedian = np.median([abs(item) for sublist in coeffs for item in sublist])
globalStd = np.std([abs(item) for sublist in coeffs for item in sublist])

H_arrays = []
horizontal_indices_seconds = []
for frequency_band in coeffs:
    thres_array, H_indices, H_array = getStdThresholds(frequency_band,max_windows=windows_count,globalMedian=globalMedian,globalStd=globalStd)
    H_arrays.append(H_array)
    horizontal_indices_seconds.append(2*np.divide(H_indices,len(frequency_band)))

minmin = 9999999
maxmax = -999999
for arr in H_arrays:
    if min(arr)<minmin: minmin = min(arr)
    if max(arr)>maxmax: maxmax = max(arr)
print(f'min and max was {minmin}, {maxmax}, calibrating max: {globalStd*4}')

frequency_counter = 0
freq_to_draw = 0
i=0
for val in level_depths:
    if frequency_counter>= freq_to_draw:
        bw = min_bandwidth*2**(level-val)
        # print(f'i: {i} val {val} for level {level} bw {bw} hurst values {H_arrays[i]} indices {horizontal_indices_seconds[i]}')
        for j in range(len(horizontal_indices_seconds[i])):
            x = horizontal_indices_seconds[i][j]
            h_val = H_arrays[i][j]
            h_val = normaliseS(h_val,globalStd)
            hurst_color = (h_val,h_val,h_val,1)
            ax.add_patch(Rectangle((x,freq_to_draw), 2, bw, color=hurst_color, fill=True, lw=lw))
        freq_to_draw += bw
        i+=1
    frequency_counter += min_bandwidth

ax.set_xlim([0,2])
ax.set_xlabel('Time (s)')
ax.yaxis.set_major_formatter(ticks)
ax.set_ylabel('Frequency (kHz)')
ax.set_title('Standard Deviation Grid')

# greyscale colorbar
# bar_ax = fig.add_subplot(122)
cmap = mpl.cm.gray
norm = mpl.colors.Normalize(vmin=50, vmax=globalStd*4)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=axs[1],ax=ax).set_label('Standard Deviation of Ỹ$_{j,k}$ [dB]')

plt.subplots_adjust(left=0.136,bottom=0.162,right=0.867,top=0.924,wspace=0.121)

############################################################################################################

## psd plot with logaritmic y axis
fig,axs=plt.subplots(1,1)
ax=axs
fig.set_size_inches(5,3)

(f,S)=scipy.signal.periodogram(recording.astype(np.float64),fs,scaling='density',return_onesided=True)

ax.plot(f,S,color='black')

ax.set_yscale('log')

ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.set_title('Power Spectral Density')
ax.xaxis.set_major_formatter(ticks)
plt.subplots_adjust(left=0.136,bottom=0.162,right=0.867,top=0.924,wspace=0.121)
plt.xlim((0,96000/2))
plt.ylim(bottom=10e-6,top=10e+3)


############################################################################################################

fig,axs=plt.subplots(1,1)
ax=axs
fig.set_size_inches(5,3)
scaled_decomp_levels, decomp_levels = getScaledDecompositionLevels(recording,max_level=level)
ax.plot(decomp_levels,color='black',linestyle='-',marker='x')
ax.set_xlabel('Decomposition level j')
ax.set_ylabel('Decomposition depth')
ax.set_title('PSD Scaled Decomposition Levels')
plt.subplots_adjust(left=0.136,bottom=0.162,right=0.867,top=0.924,wspace=0.121)


plt.show()