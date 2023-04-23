## graphing a big comparison:
## Large grid:
    ## metrics:
    ##  SnNR
    ##  SR
    ##  SDI

    ## styles:
    ##  multi-level
    ##  full
    ##  scaled
## Internal grid:
    ## x: levels
    ## y: value
    ## z: intensity

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

generic_data_array = []

# defining paths
sounds_path = '..\sounds'
possum_path = os.path.join(sounds_path,'Possum T. vulpecula 2s')
cat_path = os.path.join(sounds_path,'Cat F. Catus 2s')
munually_cleaned_path = os.path.join(sounds_path,'Manually Cleaned')
artificial_noise_added_path = os.path.join(sounds_path,'Artificial Noise')
data_out_path = './wavelet_results.xlsx'

# define which animal sounds to use
sound_file_path = artificial_noise_added_path

# load sounds
sound_clips = []
for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
    if not filename in ['.DS_Store','times.txt']:
        pathname = os.path.join(sound_file_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        sound_clips.append(signal.astype(object))

# select settings
technique = 'scaled_res'
technique = 'multi_res'
technique = 'full_res'
mother_wavelet = 'sym20'
threshold_selection = 'std_scaled'
windows = 5
level = 5
intensity = [0,5]
metric = 'frequency'

snnr_pre = []
sr_pre = []
if metric=='time':
    timesPath=os.path.join(sound_file_path,'times.txt')
    # load signal and noise timestamps
    timesArray=[]
    with open(timesPath,'r') as times:
        for line in times:
            txtvals = line[:-1].split(' ')
            floatvals = []
            for val in txtvals:
                floatvals.append(float(val))
            timesArray.append(floatvals)
    
    #calculate pre - snnr
    for i, clip in enumerate(sound_clips):
        [sigstart, sigend, noisestart, noiseend] = timesArray[i-1]
        noise_region_pre = signal[floor(fs*noisestart):floor(fs*noiseend)]
        signal_region_pre = signal[floor(fs*sigstart):floor(fs*sigend)]
        snnr_pre.append(SnNR(signal_region_pre,noise_region_pre))
        sr_pre.append(noise_region_pre)
elif metric=='frequency':
    clean_clips = []
    for i, filename in enumerate(sorted(os.listdir(munually_cleaned_path))):
        if not filename=='.DS_Store':
            pathname = os.path.join(munually_cleaned_path,filename)
            [fs, signal] = wavfile.read(pathname)
            signal = signal[:fs*2]
            clean_clips.append(signal.astype(object))

# loop over all levels
for level in range(1,10):
    for intensity in [[0,3],[0,5],[0,7]]:
        print(f'level {level}, intensity {intensity}')

        # get averaged levels
        if technique=='scaled_res':
            scaled_levels,all_levels = getScaledDecompositionLevels(sound_clips,level)
            print(f'at max level {level}, scaled levels are: {scaled_levels}')
        else: scaled_levels = 0

        # denoise clips
        denoised_clips = []
        print('denoising clips: ',end='')
        for i, clip in enumerate(sound_clips):
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

        # get metrics
        if metric=='time':
            print('getting SnNR and SR...',end=' ')
            snnr_list = []
            sr_list = []
            for i,clip in enumerate(denoised_clips):
                [sigstart, sigend, noisestart, noiseend] = timesArray[i-1]
                noise_region_post = denoised_clips[i][floor(fs*noisestart):floor(fs*noiseend)]
                signal_region_post = denoised_clips[i][floor(fs*sigstart):floor(fs*sigend)]
                snnr_post = SnNR(signal_region_post,noise_region_post)
                snnr_improvement = snnr_post - snnr_pre[i]
                snnr_list.append(snnr_improvement)
                sr_list.append(SuccessRatio(sr_pre[i],noise_region_post))
            print(f'average snnr improvement: {np.mean(snnr_list)}, average sr: {np.mean(sr_list)}')
            generic_data_array.append([[np.mean(snnr_list),np.std(snnr_list)],[np.mean(sr_list),np.std(sr_list)]])
        elif metric=='frequency':
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