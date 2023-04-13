# the perfect wavelet denoising method for bioacoustic signals!

# what is the frequency range of interest / how is spectral density distributed ?
# frequency properties gives the desired decomposition depth 
#    (or use spectral density to make spectrally scaled decomposition for perfect results)
# decomposition characteristics determine the ideal thresholding method / intensity
# and the length of the filter to use?
#   this ^,^^ relationship yet to be determined

from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels
from graphing import plot_specgram, plot_hurst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi, floor
np.set_printoptions(precision=2, suppress=True)


# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = np.random.uniform(low=10, high=20, size=(100, 100))

# Create a contour plot with a grayscale colorbar
plt.contourf(x, y, z)

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='horizontal', label='Some Units')

# Add labels and a title
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Grayscale Colorbar Example')


# frequency = 0.3
# frequency_2 = 0.1
# fs = 100
# max_level = 4
# t_start = 0
# t_end = 2
# fs = 96000

# time = np.linspace(t_start,t_end,(t_end-t_start)*fs)
# amplitude = np.sin(time*frequency*2*pi) +np.sin(time*frequency_2*2*pi)

# [hurst_vals, hurst_indices] = getHurst(amplitude,window_length_seconds=1,fs=fs,overlap=True)

# plot_hurst([hurst_vals,hurst_indices])



# plot_specgram(amplitude,fs=fs)

# fig1=plt.figure()
# fig1.subplots(2,1)
# ax1 = fig1.add_subplot(211)
# (Pxx, freqs, line) = ax1.psd(amplitude,Fs=fs,return_line=True,color='green')
# x,y=line[0].get_data()
# unaliased_amplitude = np.max(y[:2*floor(len(y)/5)])
# print(unaliased_amplitude)


# goodWavelets = ['haar','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20','coif1','coif2','coif3','coif4','coif5','meyer']
# goodWavelets = ['sym9','sym15','sym16','sym17','sym18','sym19','sym20']
# goodWavelets = ['haar','meyer','db4','db6','db10','db14','sym6','sym8','sym10','sym12','sym16','sym20','coif2','coif4']
# goodWavelets = ['meyer']
# intensities = [2,1.5,1,0.5,0,-0.5,-1,-2]
# intensities = [4]

# alias_vals = ['']*len(goodWavelets)

# ax2 = fig1.add_subplot(212)
# ax2.plot(x[4*floor(len(y)/5):],y[4*floor(len(y)/5):])

# for i,mother_wavelet in enumerate(goodWavelets):
#     print(str(mother_wavelet))
#     for intensity in intensities:
#         amplitude_denoised = denoise(amplitude,level=1,mother_wavelet=mother_wavelet,threshold_selection='test_aliasing_2',intensity=intensity)
#         (Pxx, freqs, line) = ax1.psd(amplitude_denoised,Fs=fs,return_line=True)
#         x,y=line[0].get_data()
#         aliased_amplitude = np.max(y[:2*floor(len(y)/5)])
        # print(str(mother_wavelet)+' '+str(aliased_amplitude))
        # print(str(aliased_amplitude-unaliased_amplitude))
        # alias_vals[i] = aliased_amplitude-unaliased_amplitude
        # ax2.plot(x[4*floor(len(y)/5):],y[4*floor(len(y)/5):])

# print('aliased - original:')
# for i in range(len(alias_vals)):
#     print(goodWavelets[i],end=' ')
#     print(alias_vals[i])

# goodWavelets.insert(0,'original')
# ax1.legend(goodWavelets)
# ax2.legend(goodWavelets)
# ax1.legend(['original signal','signal reconstructed with haar wavelet'])

# for mother_wavelet in getAllWavelets():
#     amplitude_denoised = denoise(amplitude,level=1,mother_wavelet=mother_wavelet,threshold_selection='test_aliasing')
#     (Pxx, freqs) = plt.psd(amplitude_denoised)
#     print(mother_wavelet,end=' ')
#     print(np.max(Pxx[:floor(len(Pxx)/2)]),end=' ')
#     print(np.max(Pxx[floor(len(Pxx)/2):]))

# amplitude_denoised = denoise(amplitude,level=1,mother_wavelet='sym20',threshold_selection='test_aliasing')


# plt.figure()
# plt.specgram(amplitude_denoised,Fs=1,mode='magnitude')
# plt.specgram(amplitude_denoised,Fs=1,scale='dB')
# plt.figure()
# plt.specgram(amplitude,Fs=1,mode='magnitude')
# plt.specgram(amplitude,Fs=1,scale='dB')
# plot_specgram(amplitude,fs=1,title='original', x_label='time (s)', y_label='frequency (Hz)', fig_size=(5,3))
# plot_specgram(amplitude_denoised,fs=1,title='original', x_label='time (s)', y_label='frequency (Hz)', fig_size=(5,3))

plt.show()