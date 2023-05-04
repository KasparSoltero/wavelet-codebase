import os
from scipy.io import wavfile
from graphing import *
from math import pi, floor
from time import time
from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
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

# define which animal sounds to use
sound_file_path = anuran_path
sound_recordings = []

### Opens selected files in order 00 - 01
for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
    ### which files to see:
    if not filename in ['.DS_Store', 'times.txt']:

        ### load file & data
        pathname = os.path.join(sound_file_path,filename)
        print(pathname)
        try: 
            [fs, signal] = wavfile.read(pathname)
        except ValueError:
            
            signal = AudioSegment.from_file(pathname, format='m4a')
            fs = signal.frame_rate
            signal = np.array(signal.get_array_of_samples())
        signal = signal[:fs*2] #make sure all files are 2 seconds exactly
        # print('{:<20},{:<20}'.format(filename,len(signal)))
        signal=signal.astype(object) #allows numpy to be more precise?

        sound_recordings.append(signal)

## get decomposition levels for scaled decomposition

recording = sound_recordings[0]
# for technique in ['multi_res','full_res','scaled_res']:
for technique in ['scaled_res']:
    print(technique)
    for level in range(1,12):
        
        if technique=='scaled_res':
            levels,level_depths = getScaledDecompositionLevels(sound_recordings,max_level=level)
            # print(levels)
        else: levels=0

        # get time in smallest measurement unit
        start_time = time()

        for i in range(1000):
            amplitude_denoised = denoise(
                recording,
                level=level,
                mother_wavelet='haar',
                technique=technique,
                threshold_selection='constant',
                levels=levels
                )
        end_time = time()

        print(f'{level} {(end_time-start_time)/1000}')
