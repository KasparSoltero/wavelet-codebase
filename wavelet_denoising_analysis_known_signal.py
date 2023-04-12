from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.io import wavfile
import os
from copy import copy
from math import floor
from waveletfuncs import getFilters, PSNR, denoise, dwt, idwt, threshold, multires, dwt2, idwt2, SNR, SnNR, generateSignal

# # getting default sampling rate and frequency info:
# [fs, signal] = wavfile.read('test.wav')
# print(fs)
# plt.specgram(signal,Fs=fs)
# plt.show()

dataOutPath='results/data_known_signal_'+str(floor(time.time()))+'.txt'

cmap = 'inferno'

# get noise
soundsPath = '/Users/kaspar/Downloads/Sounds'
noisePath = os.path.join(soundsPath,'Noise 2s')
for i, filename in enumerate(sorted(os.listdir(noisePath))):
    if (not filename in ['.DS_Store']) and (i==2):
        pathname = os.path.join(noisePath,filename)

        print('')
        print(filename)

        [fs, noise] = wavfile.read(pathname)

# get signal
sr = 96000 #Hz
frequency = 10000 #Hz
amplitude = 1000
signal = [0]*len(noise)
signal[int(len(noise)/4):3*int(len(noise)/4)] = generateSignal(length_in_seconds=1,frequency=frequency,sampling_rate=sr,amplitude=amplitude)

# combine noise and signal
noisy_signal = noise+signal
# for i in range(len(signal)):
#     noisy_signal[i+int(len(noisy_signal)/4)] += signal[i]
noisy_signal_middle = noisy_signal[int(len(noisy_signal)/4):int(len(noisy_signal)/4)+len(signal)]

plt.figure()
plt.specgram(noisy_signal,Fs=sr,cmap=cmap)

# SNR evaluation
preSNR = SNR(noisy_signal_middle,noise)
print(preSNR)

# denoise
dataOutArray = ['custom_thresholding_method_varying_intensity']
level = 10
mother_wavelet = 'db6'
threshold_selection = 'custom'
hardsoft = 'hard'
intensity = 0.4
intensities = np.linspace(0,2,20,endpoint=False)

for intensity in [0.4,0.5,0.8,0.9]:

    enhanced = denoise(noisy_signal,
                    level=level,
                    mother_wavelet=mother_wavelet,
                    threshold_selection=threshold_selection,
                    threshold_method=hardsoft,
                    thres=100, 
                    intensity=intensity)
    enhanced = enhanced.astype(object)

    plt.figure()
    plt.specgram(enhanced,Fs=sr,cmap=cmap)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('intensity '+str(intensity))

    enhanced_middle = enhanced[int(len(noisy_signal)/4):int(len(noisy_signal)/4)+len(signal)]
    enhanced_noisearea = enhanced[:int(len(noisy_signal)/4)]

    posSNR = SNR(enhanced_middle,enhanced_noisearea)
    print('intensity '+str(intensity)+' SNR improvement '+str(posSNR-preSNR))
    PSNR_value = PSNR(signal,enhanced)
    print('... psnr: '+str(PSNR_value))
# dataOutArray.append([intensity,posSNR-preSNR])

for threshold_selection in ['multi-level','minimax','universal']:
    enhanced = denoise(noisy_signal,
                        level=level,
                        mother_wavelet=mother_wavelet,
                        threshold_selection=threshold_selection,
                        threshold_method=hardsoft,
                        thres=100, 
                        intensity=intensity)
    enhanced = enhanced.astype(object)
    plt.figure()
    plt.specgram(enhanced,Fs=sr,cmap=cmap)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(threshold_selection)

    enhanced_middle = enhanced[int(len(noisy_signal)/4):int(len(noisy_signal)/4)+len(signal)]
    enhanced_noisearea = enhanced[:int(len(noisy_signal)/4)]

    posSNR = SNR(enhanced_middle,enhanced_noisearea)
    print('thresholding '+str(threshold_selection)+' SNR improvement '+str(posSNR-preSNR))
    PSNR_value = PSNR(signal,enhanced)
    print('... psnr: '+str(PSNR_value))

print(dataOutArray)

# with open(dataOutPath,'w') as dataOut:
#     for line in dataOutArray:
#         dataOut.write(str(line))

plt.show()