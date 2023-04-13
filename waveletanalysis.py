from math import sqrt, copysign, log2, floor, ceil
from tokenize import String
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve as sciconvolve
from copy import deepcopy
from waveletfuncs import getFilters, denoise, dwt, idwt, threshold, multires, dwt2, idwt2, SNR, SnNR
import time
from scipy.io.wavfile import write
import os
import math
from json.encoder import INFINITY


dataOutPath='results/data '+str(floor(time.time()))+'.txt'
soundsPath = '/Users/kaspar/Downloads/Sounds'
possumPath = os.path.join(soundsPath,'Possum T. vulpecula 2s')
catPath = os.path.join(soundsPath,'Cat F. Catus 2s')

allMotherWavelets = ['haar','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']

def getMaxLevel(signal):
 return floor(log2(len(signal)))

# define which animal sounds to use
sound_file_path = possumPath
timesPath=os.path.join(possumPath,'times.txt')

timesArray=[]
with open(timesPath,'r') as times:
    for line in times:
        txtvals = line[:-1].split(' ')
        floatvals = []
        for val in txtvals:
            floatvals.append(float(val))
        timesArray.append(floatvals)

print(timesArray)

dataOutArray=['thresholding-method possum, median improvement, std improvement, level, numinf']
thresholding_selection_methods = ['universal', 'minimax', 'multi-level']

for thresmethod in thresholding_selection_methods:
    for hardsoft in ['hard', 'soft']:
        SnNRArray = []
        print([thresmethod, hardsoft])

        for level in range(1, 20):
            print('level is ' + str(level))
            for i, filename in enumerate(sorted(os.listdir(possumPath))):
                if not filename in ['.DS_Store', 'times.txt']:
                    pathname = os.path.join(possumPath,filename)

                    print('')
                    print(filename)

                    [fs, signal] = wavfile.read(pathname)
                    signal = signal[:fs*2]

                    # print(timesArray[i-1])
                    [sigstart, sigend, noisestart, noiseend] = timesArray[i-1]
                    signal=signal.astype(object)
                    noise = signal[floor(fs*noisestart):floor(fs*noiseend)]
                    sig = signal[floor(fs*sigstart):floor(fs*sigend)]
                    preSnNR = SnNR(sig, noise)

                    enhanced = denoise(signal,level=level,mother_wavelet='db6',threshold_selection=thresmethod,threshold_method=hardsoft)
                    enhanced = enhanced.astype(object)

                    noise = enhanced[floor(fs*noisestart):floor(fs*noiseend)]
                    sig = enhanced[floor(fs*sigstart):floor(fs*sigend)]
                    posSnNR = SnNR(sig, noise)

                    print('signal to noise ratios:')
                    print(preSnNR)
                    print(posSnNR)

                    dif = posSnNR-preSnNR
                    SnNRArray.append(dif)

                    # wavfile.write('test.wav',fs,signal)
                    # wavfile.write('test_enhanced.wav',fs,enhanced)

                    # plt.specgram(signal)
                    # plt.show()
                    # plt.specgram(enhanced)
                    # plt.show()
        
            numinf = 0
            for val in SnNRArray:
                if val==INFINITY:
                    numinf+=1

            print(SnNRArray)
            dataOutArray.append(thresmethod + ' ' + hardsoft)
            dataOutArray.append(level)
            dataOutArray.append(np.median([i for i in SnNRArray if i<100000000]))
            dataOutArray.append(np.std([i for i in SnNRArray if i<100000000]))
            dataOutArray.append(level)
            dataOutArray.append(numinf)

print(dataOutArray)

# with open(dataOutPath,'w') as dataOut:
#     for line in dataOutArray:
#         dataOut.write(line)



# # evaluating methods for SNR etc
# soundfolder = '/Users/kaspar/Downloads/Sounds/'

# sArray = np.empty((len(os.listdir(soundfolder)),96000*2))
# times = os.path.join(soundfolder,'times.txt')
# timesArray = []
# with open(times, 'r') as file:
#     for line in file:
#         txtvals = line[:-1].split(' ')
#         floatvals = []
#         for val in txtvals:
#             floatvals.append(float(val))
#         timesArray.append(floatvals)
# print(len(timesArray))

# # for mother in ['haar','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']:
# # for level in range(1,max_level):
# for level in range(5,6):
#     preSnNRs = []
#     posSnNRs = []
    
#     for i, filename in enumerate(sorted(os.listdir(soundfolder))):
#         # print(filename)
#         if i==16:
#             pathname = os.path.join(soundfolder,filename)

#             try: 
#                 [fs, signal] = wavfile.read(pathname)
#                 signal = signal[:fs*2]

#                 [sigstart, sigend, noisestart, noiseend] = timesArray[i-1]
#                 noise = signal[floor(fs*noisestart):floor(fs*noiseend)]
#                 sig = signal[floor(fs*sigstart):floor(fs*sigend)]
#                 preSnNR = SnNR(sig, noise)
#                 preSnNRs.append(preSnNR)

#                 # signal=signal[:floor(fs*1)]
#                 coeffs = multires(signal,4,lpf_D,hpf_D)
#                 # freqs = [fs/8,fs/16,0,fs/4]
#                 # heights = [fs/8,fs/16,fs/16,fs/4]
#                 # fig, ax = plt.subplots()
#                 # for i,coef in enumerate(coeffs):
#                 #     for j, co in enumerate(coef):
#                 #         color=(i/len(coeffs),j/len(coef),0)
#                 #         ax.add_patch(Rectangle((np.linspace(0,1,len(coef))[j],floor(freqs[i])),i/fs,heights[i],color=color))
#                 # ax.set_xlim(xmin=0,xmax=1)
#                 # ax.set_ylim(ymin=0,ymax=fs/2)
#                 # plt.show()

#                 fig, axes = plt.subplots(4)
#                 axes[0].plot(coeffs[3])
#                 axes[1].plot(coeffs[0])
#                 axes[2].plot(coeffs[1])
#                 axes[3].plot(coeffs[2])
#                 # plt.plot(np.linspace(0,2,len(signal)),signal)
#                 # plt.specgram(signal,Fs=fs)
#                 # plt.ylabel('amplitude')
#                 # plt.xlabel('time (s)')
#                 plt.show()

#                 enhanced = denoise(signal,level,mother_wavelet='db6',threshold_selection='minimax',threshold_method='soft')

#                 noise = enhanced[floor(fs*noisestart):floor(fs*noiseend)]
#                 sig = enhanced[floor(fs*sigstart):floor(fs*sigend)]
#                 posSnNR = SnNR(sig, noise)
#                 posSnNRs.append(posSnNR)

#                 plt.specgram(enhanced)
#                 plt.show()

#             except Exception as e: 
#                 x=1
#                 # print(i)
#                 # print(repr(e))

#     # print(f'wavelet {mother}', end=', ')
#     print(f'level {level}', end=', ')
#     print(np.average(preSnNRs), end=', ')
#     print(np.std(preSnNRs), end=', ')
#     # print('post')
#     meanvals = []
#     infcount=0
#     for val in posSnNRs:
#         if val>100000000:
#             infcount+=1
#         else:
#             meanvals.append(val)
#     print(np.average(meanvals), end=', ')
#     print(np.std(meanvals), end=', ')
#     print(infcount, end=', ')
#     # print('dif')
#     difarray = np.array(posSnNRs)-np.array(preSnNRs)
#     print(np.average([i for i in difarray if i<100000000]), end=', ')
#     print(np.std([i for i in difarray if i<100000000]))

# print(original)
# print(type(original[0]))


# enhanced = enhanced.astype(np.int16)

# noise = enhanced[floor(fs*noisestart):floor(fs*noiseend)]
# sig = enhanced[floor(fs*sigstart):floor(fs*sigend)]
# postSnNR = SnNR(sig, noise)
# print(postSnNR)

# write('test.wav', fs, enhanced)
print('ended')
# print(SNR(original,enhanced))
# print(SNR(original,original))


# # Timing each level individually
# cA = original
# for i in range(10):
#     times = []
#     for k in range(2000000):
#         start = time.time()
#         [cAnew, cD] = dwt(cA, lpf_D, hpf_D)
#         stop = time.time()
#         times.append(stop-start)
#     print(f'level is {i+1}, length is {len(cAnew)}')
#     print([np.mean(times), np.std(times)])
#     cA = cAnew

# # Trying faster convolution
# arr1 = [3,2,1]
# arr2 = [1,2,3,4,5,6]

# def convolve(a1,a2):
#     # a1 filter, a2 signal
#     # zeros padding
#     n = len(a1)
#     m = len(a2)

#     out = [0]*(m+n-1)
#     for i in range(m+n-1):
#         # print(f'i is {i}')
#         # print(f'from {max(0,i-(m-1))} to {min(i,n-1)}')
#         for j in range( max(0,i-(m-1)) , min(i,n-1)+1 ):
#             # print(a1[j]*a2[i-j])
#             out[i] += a1[j]*a2[i-j]

#     return out
# start = time.time()
# out = convolve(arr1,arr2)
# stop = time.time()
# duration = stop-start
# print(duration)

# start = time.time()
# out = np.convolve(arr1,arr2)
# stop = time.time()
# duration = stop-start
# print(duration)

# start = time.time()
# out = sciconvolve(arr1,arr2)
# stop = time.time()
# duration = stop-start
# print(duration)

# # Decomposition level time tests
# levels = []
# for level in range(max_level):
#     print(f'level is {level}')
#     times = []
#     for i in range(1000000):
#         start = time.time()
#         coeffArray = multires(original,level,lpf_D,hpf_D)
#         end = time.time()
#         times.append(end-start)
#     levels.append([np.mean(times),np.std(times)])
#     print([np.mean(times),np.std(times)])

# print(levels)

# original = original[:1000]

# Mother wavelet time tests #########################################
# results = []
# for mother in ['db6','db7']#['haar','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']:
#     [lpf_R,lpf_D,hpf_R,hpf_D] = getFilters(mother)
#     times = []
#     for i in range(5000000):
#         start = time.time()
#         [ca, cd] = dwt(original, lpf_D,hpf_D)
#         end = time.time()
#         times.append(end-start)

#     print(f'wavelet {mother}, length {len(lpf_R)}')
#     mean = np.mean(times)
#     print([mean,np.std(times)])
#     results.append(mean)
# print(results)


# fig, axes = plt.subplot_mosaic('A;B;C')
# [cA,cD] = dwt(original, lpf_D, hpf_D)
# cA = threshold([cA], threshold_selection='constant',thres=100)
# axes['A'].plot(cA[0])
# axes['A'].set_title('Threshold = 100')
# axes['A'].plot([100]*len(cA[0]), 'r-')
# [cA,cD] = dwt(original, lpf_D, hpf_D)
# cA = threshold([cA], threshold_selection='constant',thres=300)
# axes['B'].plot(cA[0])
# axes['B'].set_title('Threshold = 300')
# axes['B'].plot([300]*len(cA[0]), 'r-')
# [cA,cD] = dwt(original, lpf_D, hpf_D)
# cA = threshold([cA], threshold_selection='constant',thres=500)
# axes['C'].plot(cA[0])
# axes['C'].set_title('Threshold = 500')
# axes['C'].plot([500]*len(cA[0]), 'r-')
# axes['C'].set_xlabel('Samples')
# axes['C'].set_ylabel('Amplitude')
# plt.tight_layout(pad=0.8)
# plt.show()



# count=0
# count2=0
# for i in range(len(decomposedArray)):
#     for k in range(len(decomposedArray[i])):
#         if decomposedArray[i][k]<=300:
#             count+=1
#         else:
#             count2+=1
# print(count)
# print(count2)

# decomposedArray = wpd(original,5,lpf_D,hpf_D)

# print('length of decomposed is ' + str(len(decomposedArray)))

# tempArray = []

# for i in range(len(decomposedArray)):
#     sd = np.std(decomposedArray[i])
#     print('stdev is ' + str(sd))
#     thres = 0.2*sd
#     thresholdedArray=threshold(deepcopy(decomposedArray),thres)
#     tempArray.append(thresholdedArray[i])
#     count = 0
#     for h in thresholdedArray[i]:
#         if h==0:
#             count+=1
#     print('count percent is ' + str(count/len(thresholdedArray[i])))

# reconstructed = reconstruct(original,tempArray,lpf_R, hpf_R)
# plt.figure()
# plt.specgram(reconstructed)

# plt.figure()

# for thres in [15,8,5]:

#     thresholdedArray = threshold(deepcopy(decomposedArray), thres)

#     counts = []
#     for level in thresholdedArray:
#         count = 0
#         for i in level:
#             if i==0:
#                 count+=1
#         counts.append(100*count/len(level))

#     freqranges = []
#     topf = 8000
#     for i in range(len(thresholdedArray)):
#         if i == (len(thresholdedArray)-1):
#             freqranges.append(str(topf)+' - 0')
#         else:
#             freqranges.append(str(topf)+' - ' + str(int(topf/2)))
#             topf = int(topf/2)

#     x = 0.5+np.arange(len(thresholdedArray))
#     plt.bar(x,counts)
#     plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5],freqranges)

# plt.xlabel('Frequency ranges (Hz)')
# plt.ylabel("Percent removed values")
# plt.legend(['Thres: 15','Thres: 8','Thres: 5'])
# plt.ylim([0,100])

# plt.figure()
# ax = plt.subplot(4,1,1)
# ax.specgram(original, Fs=fs)
# ax.title.set_text('Original')
# ax.get_xaxis().set_visible(False)

# decomposedArray = wpd(original,max_level,lpf_D,hpf_D)

# plt.figure()
# ax = plt.subplot(2,1,1)
# modifiedthresArray1 = deepcopy(thresholdedArray)
# modifiedthresArray2 = deepcopy(thresholdedArray)
# modifiedthresArray1[0] = decomposedArray[0]
# modifiedthresArray2[len(modifiedthresArray2)-1] = decomposedArray[len(decomposedArray)-1]
# reconstructed0 = reconstruct(original, decomposedArray, lpf_R, hpf_R)
# reconstructed1 = reconstruct(original, modifiedthresArray1, lpf_R, hpf_R)
# reconstructed2 = reconstruct(original, modifiedthresArray2, lpf_R, hpf_R)
# ax.specgram(reconstructed1, Fs=fs)
# ax.title.set_text('Highest level unchanged')
# ax = plt.subplot(2,1,2)
# ax.specgram(reconstructed2, Fs=fs)
# ax.title.set_text('Lowest level unchanged')

# plt.figure()
# ax = plt.subplot(4,1,1)
# ax.specgram(original, Fs=fs)
# ax.title.set_text('Original')
# ax.get_xaxis().set_visible(False)

# for i in range(3):
#     thres = [5,8,15]
#     thres = thres[i]
#     decomposedArray = wpd(original,max_level,lpf_D,hpf_D)
#     thresholdedArray = threshold(decomposedArray,thres)
#     reconstructed = reconstruct(original,thresholdedArray,lpf_R, hpf_R)
#     ax = plt.subplot(4,1,i+2)
#     ax.specgram(reconstructed, Fs=fs)
#     ax.title.set_text('Thres = '+str(thres))
#     if i!=2: ax.get_xaxis().set_visible(False)

# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')