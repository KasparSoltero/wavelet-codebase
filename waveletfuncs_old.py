from math import sqrt, floor, ceil, copysign, log
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

# In the Fast Wavelet Transform, the full wavelet is discretised into a set of coefficients
# This set of coefficients is referred to as the 'scaling filter'
def getCoefficients(mother_wavelet):
    # getCoefficients manually retrieves the scaling filter coefficients 
    # for the chosen mother wavelet. These coefficients can be found at:
    # http://wavelets.pybytes.com/wavelet/db1/

    match(mother_wavelet):
        case 'haar':
            coefficients = np.array([1/sqrt(2),1/sqrt(2)])
        case 'daubechies 2'|'db2':
            coefficients = np.array([0.48296291314469025,0.836516303737469,0.22414386804185735,-0.12940952255092145])
        case 'daubechies 3'|'db3':
            coefficients = np.array([0.035226291882100656,-0.08544127388224149,-0.13501102001039084,0.4598775021193313,0.8068915093133388,0.3326705529509569])
        case 'daubechies 4'|'db4':
            coefficients = np.array([-0.010597401784997278,0.032883011666982945,0.030841381835986965,-0.18703481171888114,-0.02798376941698385,0.6308807679295904,0.7148465705525415,0.23037781330885523])
        case _:
            # Incase a wavelet is used which hasn't been set up
            print('no coefficients for wavelet')

    return coefficients

# Using the scaling filter coefficients, a set of four filters is produced
def getFilters(mother_wavelet):
    # getFilters returns four filters using the process outlined in:
    # https://www.mathworks.com/help/wavelet/ug/fast-wavelet-transform-fwt-algorithm.html?ue

    # Scaling filter coefficients are retrieved from getCoefficiets:
    coefficients = getCoefficients(mother_wavelet)

    # The low pass reconstruction filter is calculated by normalising the scaling filter
    lpf_R = coefficients/np.linalg.norm(coefficients)
    
    # Initialising the high pass reconstruction filter
    hpf_R = np.array(lpf_R)
    filter_length = len(lpf_R)
    
    # The high pass reconstruction filter is a 'quadrature mirror' of the low pass reconstruction,
    # This is defined as:
    for k in range(filter_length):
        hpf_R[k] = ((-1)**k) * lpf_R[filter_length+1-k-2]

    # Low and high pass decomposition filters are 'reversed' from the reconstruction filters:
    # i.e., [ 1 , -2 ] -> [ -2 , 1 ]
    lpf_D = lpf_R[::-1]
    hpf_D = hpf_R[::-1]

    return [lpf_R,lpf_D,hpf_R,hpf_D]


def dwt(f, lpf_D, hpf_D):
    # Dwt (discrete wavelet transform) decomposes a signal into its wavelet coefficient arrays
    # Using the high and low pass decomposition filters derived from the mother wavelet
    
    # The signal is convolved with the low and high pass filters, and downsampled by 2x

    # cA, the low-pass coefficients, are referred to as 'approximate coefficients'
    cA = np.convolve(f,lpf_D)[::2] #the [::2] removes every second value / downsamples by 2

    # cD, the high-pass coefficients, are reffered to as 'detail coefficients'
    cD = np.convolve(f,hpf_D)[::2]

    # The approximate and detail coefficient arrays capture low and high frequency information respectively
    
    return [cA, cD]

# A short test signal is used to demonstrate the dwt:

# Using the db2 wavelet
mother_wavelet = 'db2'
[lpf_R,lpf_D,hpf_R,hpf_D] = getFilters(mother_wavelet)

# A sine signal with frequency f (Hz), sample rate sr (Hz), and length t (seconds)
[t,f,sr] = [0.5,10,1000]
samples = np.arange(t * sr)/sr
signal = np.sin( 2 * np.pi * f * samples)

# The signal is decomposed with dwt
[cA,cD] = dwt(signal, lpf_D, hpf_D)

# The signal and decomposed wavelet coefficient arrays are plotted:
fig, axes = plt.subplot_mosaic('AA;BC')
axes['A'].plot(signal)
axes['A'].set_title('Signal')
axes['B'].plot(cA)
axes['B'].set_title('Approximate')
axes['C'].plot(cD)
axes['C'].set_title('Detail')
axes['A'].set_xlabel('Samples')
axes['A'].set_ylabel('Amplitude')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 1 - Sine signal with single-level wavelet decomposition.
# The approximate coefficients are a copy of the signal, with half as many samples.
# The detial coefficients are very low amplitude because the signal does not contain
# high-frequency information.

# A higher frequency sine signal is added to the second half of the test signal
f2 = 200 #Hz
signal[int(len(signal)/2):] += np.sin( 2 * np.pi * f2 * samples[int(len(samples)/2):])
[cA,cD] = dwt(signal, lpf_D, hpf_D)

# The composite signal and decomposed wavelet coefficient arrays are plotted:
fig, axes = plt.subplot_mosaic('AA;BC')
axes['A'].plot(signal)
axes['A'].set_title('Signal')
axes['B'].plot(cA)
axes['B'].set_title('Approximate')
axes['C'].plot(cD)
axes['C'].set_title('Detail')
axes['A'].set_xlabel('Samples')
axes['A'].set_ylabel('Amplitude')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 2 - Composite sine signal with single-level wavelet decomposition.
# The detail coefficients contain more of the high-frequency component, while 
# the approximate coefficients contain more of the low-frequency component.

# The approximate and detail coefficients may be used to reconstruct the original signal,
# using the inverse discrete wavelet transform:
def idwt(cA,cD,match_length,lpf_R,hpf_R):
    # idwt (inverse discrete wavelet transform) 
    # Reconstructs a signal from its approximate and detail wavelet coefficient arrays
    # using the high-pass and low-pass reconstruction filters.
    
    # Zeros are inserted at every odd index, referred to as 'upsampling'
    cA2 = np.zeros(len(cA)*2-1)
    cA2[::2] = cA
    cD2 = np.zeros(len(cD)*2-1)
    cD2[::2] = cD

    # The upsampled approximate and detail coefficient arrays are convolved
    # with the reconstruction low and high pass filters respectively, and linearly summed:
    w = np.convolve(cA2,lpf_R) + np.convolve(cD2,hpf_R)

    # This process generates additional values at the edges of the signal, so these values
    # must be removed to return a signal of the same length as the original (match_length).
    reconstructed = w[ceil((len(w)-len(match_length))/2):len(w)-floor((len(w)-len(match_length))/2)]

    return reconstructed


# The detail and approximate coefficient arrays are not equivalent to the 
# signals represented by these arrays.
# In order to see the signal represented by a single approximate or detail array, 
# all other arrays are set to zero, and the signal is reconstructed from a single array 
# using the inverse discrete wavelet transform:

cA_reconstructed = idwt(cA,cD*0,signal,lpf_R,hpf_R) # Setting the detail coefficients to zero
cD_reconstructed = idwt(cA*0,cD,signal,lpf_R,hpf_R) # Setting the approximate coefficients to zero

# The signals reconstructed using only one of the approximate or detail arrays are plotted:
fig, axes = plt.subplot_mosaic('A;B;C')
axes['A'].plot(signal)
axes['A'].set_title('Signal')
axes['B'].plot(cA_reconstructed)
axes['B'].set_title('Signal from approximate coefficients')
axes['C'].plot(cD_reconstructed)
axes['C'].set_title('Signal from detail coefficients')
axes['B'].set_ylabel('Amplitude')
axes['C'].set_xlabel('Samples')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 3 - Reconstructed signals from approximate and detail coefficients.
# The amplitudes in this figure are directly comparable, and show how 
# the approximate and detail arrays extract different components of the signal.

# For more complex signals, the approximate array may be further decomposed yielding multi-resolution analysis:
def multires(signal, level, lpf_D, hpf_D):
    # multires (Multiple-resolution analysis) repeatedly decomposes the approximate coefficients
    # Because the approximate coefficients represent the low frequency part of the signal,
    # each level analyses a lower frequency portion of the signal.

    # This process of multi-resolution analysis is shown below:
    # 
    # [signal]──┬─>[cD1] ┄┄┄┄┄┄┄┄┄┄┄┄┄┄[cD1]        level 1
    #           └─>[cA1]──┬─>[cD2]┄┄┄┄┄[cD2]        level 2
    #                     └─>[cA2]──┬─>[cD3]        level 3
    #                               └─>[cA3]
    # 
    # Each decomposition splits the signal into a higher and lower frequency component,
    # so the frequencies spanned by the final sets of coefficients
    # for a signal sampled at 1000 Hz, decomposed to level 3 as above would be:
    #
    #   cD1: 1000 - 500 Hz
    #   cD2: 500 - 250 Hz
    #   cD3: 250 - 125 Hz
    #   cA3: 125 - 0 Hz

    # An array is used to store the coefficients at each resolution
    coefficientArray = []

    # For the first decomposition, the original signal is used.
    # Each successive decomposition uses the approximate coefficients of the previous level
    cA = signal

    for i in range(level):
        # At each level, the approximate coefficients are decomposed into
        # low and high frequency components using the discrete wavelet transform.
        [cA, cD] = dwt(cA, lpf_D, hpf_D)

        # Only the detail (high frequency) coefficients are stored, as the 
        # approximate coefficients are used in the following iteration.
        coefficientArray.append(cD)

    # Finally, the last approximate coefficients are added to the end of the array
    coefficientArray.append(cA)

    return coefficientArray

# The test signal is used to show how multi-resolution analysis breaks a signal into distinct components
# using a level 3 decomposition:
coefficientArray = multires(signal, 3, lpf_D, hpf_D)
fig, axes = plt.subplot_mosaic('A;B;C;D;E')
axes['A'].plot(signal)
axes['A'].set_title('Signal')
axes['B'].plot(coefficientArray[0])
axes['B'].set_title('cD 1')
axes['C'].plot(coefficientArray[1])
axes['C'].set_title('cD 2')
axes['D'].plot(coefficientArray[2])
axes['D'].set_title('cD 3')
axes['E'].plot(coefficientArray[3])
axes['E'].set_title('cA 3')
axes['C'].set_ylabel('Amplitude')
axes['E'].set_xlabel('Samples')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 4 - 3 Levels of signal decomposition using multi-resolution analysis.
# This shows how the levels capture progressively lower frequency information

# After analyzing a signal at multiple 'resolutions', an inverse transformation is made
# to reconstruct the original signal from the many decomposition levels:
def inverseMultires(signal, coefficientArray, lpf_R, hpf_R):
    # inverseMultires reconstructs the original signal from the
    # set of coefficients generated by multi-resolution analysis.

    # This process works in reverse to multi-resolution decomposition,
    # iteratively reconstructing each approximate coefficient array from the lower level:

    # [cD1]──────────────────────┬─>[signal]
    # [cD2]────────────┬─>[cA1]──┘
    # [cD3]──┬─>[cA2]──┘
    # [cA3]──┘

    # The initial approximate coefficients are the last in the coefficient array
    cA = coefficientArray[len(coefficientArray)-1]

    # The number of levels to reconstruct is determined by the length of the provided array
    for i in range(len(coefficientArray)-1):

        # Detail coefficients are set appropriately at each level
        cD = coefficientArray[(len(coefficientArray)-2)-i]

        # The inverse discrete wavelet transform must know the appropriate length of the reconstruction.
        # This is set to match the detail coefficients of the level above at each iteration,
        # and is set to match the original signal for the final reconstruction.
        if (i==(len(coefficientArray)-2)):
            match_length = signal
        else:
            match_length = coefficientArray[(len(coefficientArray)-3)-i]

        # The higher level approximate coefficients are reconstructed from the 
        # lower level detail and approximate coefficients:
        reconstructed = idwt(cA, cD, match_length, lpf_R, hpf_R)

        # Finally, the approximate coefficient is reset for the next iteration.
        cA = reconstructed

    # After all reconstructions, the fully reconstructed signal is returned
    return reconstructed

# Wavelet denoising is achieved by decomposing a signal, reducing the magnitude of 
# coefficients below a threshold magnitude in the detail arrays, and reconstucting the signal.
def threshold(coeffs, threshold_method='hard', threshold_selection='universal',thres=0):
    # threshold takes a signal decomposed using multi-resolution analysis, and 
    # thresholds every coefficient using 'hard' or 'soft' thresholding.
    # There are a variety of ways to select a threshold value, with the method specified by 'threshold_selection'.
    
    thresholds = []

    if (threshold_selection=='constant'):
        # If a constant threshold is used, this applies to every decomposition level.
        thresholds = [thres]*len(coeffs)
        print(thresholds)
    elif (threshold_selection=='universal'):
        # The most common method is referred to as 'universal' thresholding.
        # this uses the noise level and signal length to calculate a threshold for each level.
        # The noise level is approximated for each level as defined in https://doi.org/10.1088/1741-2552/abc741
        noise_levels = []
        for i in range(len(coeffs)):
            noise_level = np.median(np.abs(coeffs[i]))/0.6745
            noise_levels.append(noise_level)
    
        thresholds = np.array(noise_levels)*sqrt(2*log(len(coeffs[0])*2))

    elif (threshold_selection=='minimax'):
        # Minimax thresholding uses a similar approach to universal thresholding, 
        # however the thresholds are calculated slightly differently, shown here:
        noise_levels = []
        for i in range(len(coeffs)):
            noise_level = np.median(np.abs(coeffs[i]))/0.6745
            noise_levels.append(noise_level)

        thresholds = np.array(noise_levels)*(0.3936 + 0.1829*log(len(coeffs[0])*2,2))

    # For each of the decomposition level arrays (cA1, cA2, cA3, cD3 etc),
    # Each coefficient in the array is compared to the threshold value.
    for i in range(len(coeffs)):
        for k in range(len(coeffs[i])):
            if abs(coeffs[i][k])<thresholds[i]:
                # In both hard and soft thresholding, values less than the threshold are set to 0
                coeffs[i][k] = 0
            elif threshold_method=='soft':
                # In soft thresholding, values higher than the threshold are reduced in magnitude
                # by the threshold amount.
                coeffs[i][k] = copysign(abs(coeffs[i][k])-thresholds[i],coeffs[i][k])
    
    return coeffs

# Hard and soft thresholding is demonstrated using a test signal, with constant threshold value 0.5.
# The test signal has a frequency of 5 Hz, length of 5 seconds, and sampling rate 1000 Hz.
[t,f,sr] = [0.5,5,1000]
samples = np.arange(t * sr)/sr
signal = np.sin( 2 * np.pi * f * samples)

coeffs = multires(signal,1,lpf_D,hpf_D)
fig, axes = plt.subplot_mosaic('A;B;C')
axes['A'].plot(coeffs[1])
axes['A'].plot(coeffs[1]*0 + 0.5, 'r')
axes['A'].plot(coeffs[1]*0 - 0.5, 'r')
axes['A'].set_title('1st level approximate pre-threshold')
coeffs = threshold(coeffs,threshold_method='hard',threshold_selection='constant',thres=0.5)
axes['B'].plot(coeffs[1])
axes['B'].plot(coeffs[1]*0 + 0.5, 'r-')
axes['B'].plot(coeffs[1]*0 - 0.5, 'r-')
axes['B'].set_title('1st level approximate hard-thresholded')
coeffs = multires(signal,1,lpf_D,hpf_D)
coeffs = threshold(coeffs,threshold_method='soft',threshold_selection='constant',thres=0.5)
axes['C'].plot(coeffs[1])
axes['C'].plot(coeffs[1]*0 + 0.5, 'r')
axes['C'].plot(coeffs[1]*0 - 0.5, 'r')
axes['C'].set_title('1st level approximate soft-thresholded')
axes['B'].set_ylabel('Amplitude')
axes['C'].set_xlabel('Samples')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 5 - Hard and soft thresholding with threshold value 0.5
# Soft thresholding clearly sacrifices signal strength for smoothness.

def denoise(signal,level,mother_wavelet='haar',threshold_selection='universal',threshold_method='hard', thres=0):
    # denoise takes a signal through the entire wavelet denoising process,
    # by decomposing it with multi-resolution analysis, thresholding it, and reconstructing it.
    # The process diagram is as follows:

    #   [signal]──┬─>[cD1] ┄┄┄┄┄┄┄┄┄┄┄┄┄┄[cD1]──────────────────────┬─>[signal]
    #             └─>[cA1]──┬─>[cD2]┄┄┄┄┄[cD2]────────────┬─>[cA1]──┘
    #                       └─>[cA2]──┬─>[cD3]──┬─>[cA2]──┘
    #                                 └─>[cA3]──┘
    #                                      ^
    #                                 thresholding

    [lpf_R,lpf_D,hpf_R,hpf_D] = getFilters(mother_wavelet)

    coeffs = multires(signal,level,lpf_D,hpf_D)
    
    thresholded_coeffs = threshold(coeffs, threshold_method, threshold_selection, thres)

    denoised_signal = inverseMultires(signal,thresholded_coeffs,lpf_R,hpf_R)

    return denoised_signal

# A composite test signal of frequencies 5 and 200 Hz is used to demonstrate basic wavelet denoising:
# A constant threshold of 0.5 is used.
[t,f1,f2,sr] = [0.5,5,200,1000]
samples = np.arange(t * sr)/sr
signal = np.sin( 2 * np.pi * f1 * samples)
signal +=  np.sin( 2 * np.pi * f2 * samples)*0.3
fig, axes = plt.subplot_mosaic('A;B')
axes['A'].plot(signal)
axes['A'].set_title('Original Signal')
denoised_signal = denoise(signal, level=3, threshold_selection='constant',thres=0.5)
axes['B'].plot(denoised_signal)
axes['B'].set_title('Denoised Signal')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
axes['B'].set_xlabel('Samples')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 6 - Original and denoised composite signal
# This shows how wavelet denoising is able to reduce parts of a signal regarded as 'noise',
# while retaining the overall character of the signal.

# A real-world example containing a cat vocalisation is used to demonstrate the process:
[Fs, original] = wavfile.read('cat.wav')
fig, axes = plt.subplot_mosaic('A;B;C')
axes['A'].specgram(original,Fs=Fs)
axes['A'].set_title('Original')
universal_denoised = denoise(original,level=5,threshold_selection='universal')
axes['B'].specgram(universal_denoised,Fs=Fs)
axes['B'].set_title('Universal thresholding')
minimax_denoised = denoise(original,level=5,threshold_selection='minimax')
axes['C'].specgram(minimax_denoised,Fs=Fs)
axes['C'].set_title('Minimax thresholding')
axes['B'].set_ylabel('Frequency (Hz)')
axes['C'].set_xlabel('Time (s)')
plt.tight_layout(pad=0.8)
plt.show()

# Figure 7 - Specrograms of cat vocalisation denoised using universal and minimax thresholding.
# This shows both thresholding methods performing similarly, however methods
# must typically be adjusted to better match a specific use case. The regions between vocalisations are
# not displayed on the plot because there is insufficient non-zero values to estimate a frequency.