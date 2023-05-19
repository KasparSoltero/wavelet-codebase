
from math import sqrt, copysign, floor, ceil, log
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve as sciconvolve
from copy import deepcopy
from waveletfuncs import getAllWavelets, getMaxLevel, SuccessRatio, PSNR, getFilters, denoise, dwt, idwt, threshold, multires, dwt2, idwt2, SNR, SnNR, getScaledDecompositionLevels, getHurstThresholds, PreservationRatio, SDI
from graphing import plot_specgram, add_specgram, add_signal_noise_patch, four_specgrams, plot_hurst, scaled_bands_visual,noise_grid_visual
import time
from scipy.io.wavfile import write
import os
import math
import random

generic_data_array = []

# defining paths
sounds_path = '.\sounds'
## animals
possum_manually_cleaned = os.path.join(sounds_path,'possum_2s_manually_cleaned')
cat_manually_cleaned = os.path.join(sounds_path,'cat_2s_manually_cleaned')
bird_manually_cleaned = os.path.join(sounds_path,'bird_2s_manually_cleaned')
## get noise
noise_path = os.path.join(sounds_path,'Noise 2s')
noise_clips = []
for i, filename in enumerate(sorted(os.listdir(noise_path))):
    if not filename in ['.DS_Store']:
        pathname = os.path.join(noise_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        noise_clips.append(signal.astype(object))

## choose animal
clean_signals_path = bird_manually_cleaned
# clean_signals_path = cat_manually_cleaned
# clean_signals_path = possum_manually_cleaned

## add noise to clean signals randomly
clean_clips = []
dirty_clips = []
for i, filename in enumerate(sorted(os.listdir(clean_signals_path))):
    if not filename=='.DS_Store':
        pathname = os.path.join(clean_signals_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        if len(signal) < fs*2:
            print(len(signal))
            # signal = np.pad(signal, (0,fs*2-len(signal)), 'constant')
        else:
            clean_clips.append(signal.astype(object))
            
            random_noise = random.choice(noise_clips)
            dirty_clips.append(np.add(signal.astype(object),random_noise.astype(object)))

# munually_cleaned_path = os.path.join(sounds_path,'Manually Cleaned')
artificial_noise_added_path = os.path.join(sounds_path,'Artificial Noise')
# data_out_path = './wavelet_results.xlsx'

# define which animal sounds to use
sound_file_path = artificial_noise_added_path

# select settings
technique = 'scaled_res'
# technique = 'multi_res'
# technique = 'full_res'
mother_wavelet = 'sym20'
threshold_selection = 'std_scaled'
windows = 5
level = 5
intensity = [0,5]
metric = 'frequency'

# loop over all levels
for level in range(7,10):
    for intensity in [[0,3],[0,5],[0,7]]:
        print(f'level {level}, intensity {intensity}')

        # get averaged levels
        if technique=='scaled_res':
            scaled_levels,all_levels = getScaledDecompositionLevels(dirty_clips,level)
            print(f'at max level {level}, scaled levels are: {scaled_levels}')
        else: scaled_levels = 0

        # denoise clips
        denoised_clips = []
        print('denoising clips: ',end='')
        for i, clip in enumerate(dirty_clips):
            print(i,end=', ')
            denoised_clips.append(denoise(clip, 
                level=level,
                mother_wavelet=mother_wavelet,
                technique=technique,
                threshold_selection=threshold_selection,
                threshold_method='hard',
                storefile=False,
                fs=96000,
                levels=scaled_levels,
                windows=windows,
                intensity=intensity).astype(object))
        print('... finished denoising.')

        plt.figure()
        plt.specgram(denoised_clips[0],Fs=96000,vmin=-30,vmax=40)
        plt.title('denoised')
        plt.figure()
        plt.specgram(dirty_clips[0],Fs=96000,vmin=-30,vmax=40)
        plt.title('dirty')
        plt.figure()
        plt.specgram(clean_clips[0],Fs=96000,vmin=-30,vmax=40)
        plt.title('clean')
        plt.show()

        # get metrics
        if metric=='frequency':
            print('getting SDI...',end=' ')
            sdi_list = []
            for i,clip in enumerate(denoised_clips):
                sdi_list.append(SDI(clip,clean_clips[i]))
            print(f'average sdi: {np.mean(sdi_list)}')
            generic_data_array.append([[np.mean(sdi_list),np.std(sdi_list)],[0,0]])

print('final data array:')
# print final data array in a format which can be pasted into excel:
for j,intensity in enumerate([[0,3],[0,5],[0,7]]):
    for i,level in enumerate(range(1,10)):
        row = generic_data_array[i*3+j]
        print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(level,intensity[1],row[0][0],row[0][1],row[1][0],row[1][1]))