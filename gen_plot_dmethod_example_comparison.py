import os
from scipy.io import wavfile
from graphing import *
from math import pi, floor
from time import time
from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters, dwt, getFilters
import numpy as np
from pydub import AudioSegment

# Set paths
sounds_path = './sounds'
possum_path = os.path.join(sounds_path,'Possum T. vulpecula 2s')
cat_path = os.path.join(sounds_path,'Cat F. Catus 2s')
bird_path = os.path.join(sounds_path, 'bird_2s')
anuran_path = os.path.join(sounds_path, 'anuran_2s')
munually_cleaned_path = os.path.join(sounds_path,'Manually Cleaned')
artificial_noise_added_path = os.path.join(sounds_path,'Artificial Noise')
data_out_path = './wavelet_results.xlsx'

# define which animal sounds to use 3, 55, 30, 29
sound_file_path = bird_path
sound_recordings = []

### Opens selected files in order 00 - 01
for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
    ### which files to see:
    # if not filename in ['.DS_Store', 'times.txt'] and i<10:
    if filename=='2841d9c1-104c-431e-8b30-c86529f05bba_01.wav':

        ### load file & data
        pathname = os.path.join(sound_file_path,filename)
        print(pathname)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2] #make sure all files are 2 seconds exactly
        # print('{:<20},{:<20}'.format(filename,len(signal)))
        signal=signal.astype(object) #allows numpy to be more precise?
        sound_recordings.append(signal)

for k,recording in enumerate(sound_recordings):
# recording = sound_recordings[0]

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(5,3)
    add_specgram(recording,axs[0][0])
    axs_list = [axs[0][0],axs[0][1],axs[1][0],axs[1][1]]
    title_list = ['Original','Multi-resolution','Full-resolution','Scaled-resolution']

    for i,technique in enumerate(['multi_res','full_res','scaled_res']):
    # for technique in ['scaled_res']:
        level = 10
        mother_wavelet = 'sym20'
        reconstructed = denoise(
            recording,
            technique=technique,
            level=level,
            mother_wavelet=mother_wavelet,
            threshold_method='hard',
            threshold_selection='std_scaled',
            intensity=[0,5]
        )
        add_specgram(reconstructed,axs_list[i+1])

# kHz
for i in range(4):
    scale = 1e3                     # KHz
    ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    axs_list[i].yaxis.set_major_formatter(ticks)
    axs_list[i].set_title(title_list[i])
# turn off y ticks for the right two plots
axs_list[3].set_yticks([])
axs_list[1].set_yticks([])
# and x ticks for the top two
axs_list[0].set_xticks([])
axs_list[1].set_xticks([])

fig.text(0.03, 0.5, 'Frequency (kHz)', ha='center', va='center', rotation='vertical')
fig.text(0.51, 0.03, 'Time (s)', ha='center', va='center')
#adjust subplots
plt.subplots_adjust(left=0.107,right=0.95,bottom=0.136,top=0.926,wspace=0.088,hspace=0.252)

plt.show()