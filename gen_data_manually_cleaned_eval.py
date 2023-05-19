import os
from scipy.io import wavfile
import numpy as np
import scipy.signal
import random
from waveletfuncs import denoise, SDI
from waveletfuncs import *
from graphing import *

## add noise to clean signals randomly
clean_signals_path_bird = './Sounds/bird_2s_manually_cleaned'
clean_signals_path_possum = './Sounds/possum_2s_manually_cleaned'
artificial_noise_path = './Sounds/Artificial Noise'
clean_clips = []
dirty_clips = []

clean_signals_path = clean_signals_path_bird

for i, filename in enumerate(sorted(os.listdir(clean_signals_path))):
    if not filename=='.DS_Store':
        pathname = os.path.join(clean_signals_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        clean_clips.append(signal.astype(object))

for i, filename in enumerate(sorted(os.listdir(artificial_noise_path))):
    if not filename=='.DS_Store':
        pathname = os.path.join(artificial_noise_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        dirty_clips.append(signal.astype(object))

##
## denoise and eval
## 
        
level = 5
mother_wavelet = 'sym20'
threshold_selection = 'std_scaled'
# threshold_selection = 'custom_alt_noise_levels'
# threshold_selection = 'minimax'
# threshold_selection = 'constant'
hardsoft = 'hard'
storefile = True
technique = 'scaled_res'

scaled_decomp_levels_store_possum = [
    [1, 0],
    [2, 0, 0],
    [3, 0, 1, 0, 0],
    [4, 0, 1, 0, 2, 0, 1, 0, 0],
    [5, 0, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2, 0, 1, 0, 0],
    [6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 0, 2, 1, 1, 0, 3, 0, 1, 0, 2, 0, 0, 0],
    [7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 0, 0],
    [8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 0, 0, 2, 2, 1, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 0, 0, 0],
    [8, 1, 0, 1, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 1, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 0, 0, 0, 3, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3, 1, 0, 0, 3, 0, 0, 0, 0],
    [9, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 0, 1, 1, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 0, 0, 3, 1, 1, 1, 1, 1, 0, 2, 1, 2, 1, 0, 1, 1, 0, 4, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 1, 1, 0, 1, 2, 0, 0, 2, 2, 0, 1, 0, 3, 0, 0, 0, 8, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
]
scaled_decomp_levels_store_cat = [
    [1, 0],
    [2, 0, 0],
    [3, 0, 1, 0, 0],
    [4, 0, 0, 1, 1, 0, 0],
    [5, 0, 0, 0, 1, 2, 0, 1, 0, 0],
    [6, 0, 1, 0, 2, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 0],
    [7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 0, 0, 0, 1, 4, 0, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 0, 0],
    [7, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 5, 0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 0, 0],
    [7, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 0, 5, 0, 1, 0, 1, 1, 0, 3, 0, 0, 0, 0, 0, 1, 6, 0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 1, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 0, 4, 0, 1, 0, 1, 1, 0, 3, 0, 0, 0, 0],
    [7, 3, 0, 1, 0, 1, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 0, 5, 1, 0, 2, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0, 0, 4, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 6, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0, 1, 1, 1, 0, 3, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 0, 3, 2, 0, 1, 0, 1, 1, 1, 0, 4, 0, 1, 0, 2, 0, 0, 0, 0]
]


# for level in range(0,10):
for windows in range(5,11):
    print(f'{windows}',end=' ')

    scaled_levels = scaled_decomp_levels_store_possum[level]
    
    scaled_decomp_levels, decomp_levels = getScaledDecompositionLevels(dirty_clips,max_level=level)
    scaled_levels = scaled_decomp_levels

    SDI_vals = []

    for i,signal in enumerate(dirty_clips):
    # for i,signal in enumerate(dirty_clips[0:1]):
        
        # print(f'{level} {scaled_levels} denoising signal: {len(signal)} {type(signal)} {type(signal[0])}')
        auto_cleaned_signal = denoise(signal,
            level=level,
            mother_wavelet=mother_wavelet,
            technique=technique,
            threshold_selection=threshold_selection,
            threshold_method=hardsoft,
            storefile=storefile,
            fs=96000,
            levels=scaled_levels,
            windows=windows
        ).astype(object)

        SDI_val = SDI(auto_cleaned_signal,clean_clips[i])
        SDI_vals.append(SDI_val)
        print(f'{SDI_val}',end=' ')

    SDI_mean = np.mean(SDI_vals)
    SDI_std = np.std(SDI_vals)
    print('{:} {:} {:}'.format(windows,SDI_mean,SDI_std))

        # plot_specgram(auto_cleaned_signal)
        # plt.show()