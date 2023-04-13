from math import sqrt, copysign, floor, ceil, log
from tokenize import String
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve as sciconvolve
from copy import deepcopy
from waveletfuncs import getAllWavelets, getMaxLevel, SuccessRatio, PSNR, getFilters, denoise, dwt, idwt, threshold, multires, dwt2, idwt2, SNR, SnNR, getScaledDecompositionLevels, getHurstThresholds, PreservationRatio
from graphing import plot_specgram, add_specgram, add_signal_noise_patch, four_specgrams, plot_hurst, scaled_bands_visual,noise_grid_visual
import time
from scipy.io.wavfile import write
import os
import math
from json.encoder import INFINITY

dataOutPath='results/data '+str(floor(time.time()))+'.txt'
soundsPath = '/Users/kaspar/Downloads/Sounds'
possumPath = os.path.join(soundsPath,'Possum T. vulpecula 2s')
catPath = os.path.join(soundsPath,'Cat F. Catus 2s')

# define which animal sounds to use
sound_file_path = possumPath
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

# initialise data out array
dataOutArray=['thresholding_method_vs_SNR_evaluation_test possum_files']

#initialise temporary data arrays
thresholding_selection_methods = ['universal', 'minimax', 'multi-level']
high_SNR_files_possum = ['02.wav','15.wav','36.wav']
low_SNR_files_possum = ['05.wav','31.wav','25.wav']
high_SNR_files_cat = ['35.wav','19.wav','58.wav']
low_SNR_files_cat = ['11.wav','10.wav','00.wav']

# denoise settings
level = 5
mother_wavelet = 'sym20'
threshold_selection = 'std_scaled'
# threshold_selection = 'custom_alt_noise_levels'
# threshold_selection = 'minimax'
# threshold_selection = 'constant'
hardsoft = 'hard'
intensity = -1
intensities = np.linspace(0.5,0.7,10,endpoint=False)
# intensities = [0.5,0.4,0.3,0.2,0.1,0]
storefile = True
# technique = 'multi_res'
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


#### uncomment for all stats vals
# for mother_wavelet in getAllWavelets():
for level in range(9,10):
    SnNR_array = []
    SR_array = []
    PR_array = []
    
    scaled_decomp_levels = scaled_decomp_levels_store_possum[level-1]
    print(scaled_decomp_levels)

    ## for getting results related to initial SnNR
    # print('initial_SnNR ',end='')
    # for intensity in intensities:
    #     print("SnNR_increase_"+str(intensity)+' ',end='')
    #     print("SR_"+str(intensity)+' ',end='')
    #     print("PR_"+str(intensity)+' ',end='')
    # print('\n',end='')

    # all_signals=[]

    ### Opens selected files in order 00 - 01
    for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
        ### which files to see:
        # if not filename in ['.DS_Store', 'times.txt']:
        # if filename in high_SNR_files_possum:
        if filename=='34.wav':

            ### load file & data
            pathname = os.path.join(sound_file_path,filename)
            [sigstart, sigend, noisestart, noiseend] = timesArray[i-1]
            [fs, signal] = wavfile.read(pathname)
            signal = signal[:fs*2] #make sure all files are 2 seconds exactly
            # print('{:<20},{:<20}'.format(filename,len(signal)))
            signal=signal.astype(object) #allows numpy to be more precise?
            noise_region_pre = signal[floor(fs*noisestart):floor(fs*noiseend)]
            signal_region_pre = signal[floor(fs*sigstart):floor(fs*sigend)]


            ### Load initial mod. signal to noise ratio
            pre_SnNR = SnNR(signal_region_pre,noise_region_pre)

            # all_signals.append(signal)
            # four_specgrams(signal)
            # scaled_bands_visual(signal,level=level,fs=fs)
            # noise_grid_visual(signal,method='entropy',level=level,fs=fs,windows=10,x_label='Time (s)',y_label='Frequency (Hz)')
            # getScaledDecompositionLevels(signal,max_level=10)

            ### Print file data?
            # print(str(filename)+' '+str(pre_SnNR))

            ## Draw Original Sound
            # plot_specgram(signal,fs=fs)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Frequency (Hz)')
            # plt.title('original')
            # plt.title('SnNR: '+str(pre_SnNR))
            # plt.xlim((0,2))
            # plt.subplots_adjust(bottom=0.15, left=0.15)
            # plt.axvline(x=floor(fs*noisestart)/fs,color='red')
            # plt.axvline(x=floor(fs*noiseend)/fs,color='red')
            # plt.axvline(x=floor(fs*sigstart)/fs,color='green')
            # plt.axvline(x=floor(fs*sigend)/fs,color='green')

            #####Choose internal loop
            # for intensity in intensities:
            # for threshold_selection in thresholding_selection_methods:
            # for mother_wavelet in ['sym20']:
            # for level in [1,10]:
            # for technique in ['multi_res','scaled_res']:
                    
            ### Denoise
            
            enhanced = denoise(signal,
                    level=level,
                    mother_wavelet=mother_wavelet,
                    threshold_selection=threshold_selection,
                    threshold_method=hardsoft,
                    thres=0, 
                    intensity=intensity,
                    storefile=storefile,
                    fs=fs,
                    technique=technique,
                    levels=scaled_decomp_levels
                )
            enhanced = enhanced.astype(object)
            ## and get data
            noise_region = enhanced[floor(fs*noisestart):floor(fs*noiseend)]
            signal_region = enhanced[floor(fs*sigstart):floor(fs*sigend)]

            post_SnNR = SnNR(signal_region,noise_region)
            SnNR_improvement = post_SnNR-pre_SnNR
            SR_val = SuccessRatio(noise_region_pre,noise_region)
            Preservation_Ratio = PreservationRatio(noise_region_pre,noise_region,signal_region_pre,signal_region)

            ## for gettings results related to initial SnNR
            # print(str(SnNR_improvement)+' ',end='')
            # print(str(SR_val)+' ',end='')
            # print(str(Preservation_Ratio)+' ',end='')
            # print('\n',end='')

            # plot enhanced signal
            # plot_specgram(enhanced,fs=fs)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Frequency (Hz)')
            # plt.subplots_adjust(bottom=0.15, left=0.15)
            # plt.title('enhanced')
            # plt.title('Method '+str(threshold_selection))
            # plt.title('Intensity '+str(intensity))
            # plt.title('level '+str(level))
            # plt.axvline(x=floor(fs*noisestart)/fs,color='black')
            # plt.axvline(x=floor(fs*noiseend)/fs,color='black')
            # plt.axvline(x=floor(fs*sigstart)/fs,color='red')
            # plt.axvline(x=floor(fs*sigend)/fs,color='red')

            ######### Bottom of internal loop


            ## plot original and enhanced signals on the same graph
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            add_specgram(signal,ax=ax1,fs=fs)
            add_signal_noise_patch(ax1,[sigstart,sigend,noisestart,noiseend],fs=fs)
            add_specgram(enhanced,ax=ax2,fs=fs)
            ## add_signal_noise_patch(ax2,[sigstart,sigend,noisestart,noiseend],fs=fs)
            # # ax1.set_title('Original recording')
            # # ax2.set_title('Processed recording')
            # fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical')
            # fig.text(0.5,0.04,'Time (s)',ha='center')
            # fig.subplots_adjust(left=0.16,bottom=0.12)
            
            plt.xlim((0,2))

            print(f'SR: {SR_val}, SnNR imp. {SnNR_improvement}, PR {Preservation_Ratio}')
            # plt.show()
            
            # plt.title('method '+str(threshold_selection))
            # plt.title('intensity '+str(intensity))
            # plt.title('level '+str(level))

            ## output data, when using single file
            # print("Thres method: {:<10} SnNR increase: {:<10.4f} SR: {:<10.4f} MSE ratio: {:<10.4f}".format(threshold_selection, SnNR_improvement, SR_val, MSE_ratio))
            # print("Intensity: {:<10.4f} SnNR increase: {:<10.4f} SR: {:<10.4f} MSE ratio: {:<10.4f}".format(intensity, SnNR_improvement, SR_val, MSE_ratio))

            ### Store for stats vals
            # SnNR_array.append(SnNR_improvement)
            # SR_array.append(SR_val)
            # PR_array.append(Preservation_Ratio)

    ########## Bottom of stats vals loop

    # scaled_decomp_levels, decomp_levels = getScaledDecompositionLevels(all_signals,max_level=level)
    # print('scaled_decomp_levels:')
    # print(scaled_decomp_levels)

    ### Calculate stats
    # SnNR_mean=np.mean(SnNR_array)
    # SnNR_std = np.std(SnNR_array)
    # SR_mean=np.mean(SR_array)
    # SR_std = np.std(SR_array)
    # PR_mean=np.mean(PR_array)
    # PR_std = np.std(PR_array)

    #### Output stats
    # print("level {:<10} SnNR_increase {:<10} SnNR_std {:<10} SR {:<10.4f} SR_std {:<10} PR {:<10} PR_std {:<10}".format(level, SnNR_mean, SnNR_std, SR_mean, SR_std, PR_mean, PR_std))
    # print("{:} {:} {:} {:} {:} {:} {:}".format(level, SnNR_mean, SnNR_std, SR_mean, SR_std, PR_mean, PR_std))
    # [lpf_R,lpf_D,hpf_R,hpf_D] = getFilters(mother_wavelet)
    # print("wavelet {:<10} length {:<10} SnNR_increase {:<10} SnNR_std {:<10} SR {:<10.4f} SR_std {:<10} MSE_ratio {:<10} MSE_std {:<10}".format(mother_wavelet, len(lpf_R), SnNR_mean, SnNR_std, SR_mean, SR_std, MSE_mean, MSE_std))

# print(dataOutArray)
plt.show()

# with open(dataOutPath,'w') as dataOut:
#     for line in dataOutArray:
#         dataOut.write(line)