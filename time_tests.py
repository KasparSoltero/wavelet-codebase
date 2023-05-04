from waveletfuncs import denoise, getAllWavelets, getScaledDecompositionLevels, getFilters
import numpy as np
from math import pi, floor
from time import time

frequency = 10
fs = 96000
t_start = 0
t_end = 2

time_array = np.linspace(t_start,t_end,(t_end-t_start)*fs)
amplitude = np.sin(time_array*frequency*2*pi)

# for technique in ['multi_res','full_res','scaled_res']:
for technique in ['scaled_res']:
    print(technique)
    for level in range(1,20):
        
        if technique=='scaled_res':
            levels,level_depths = getScaledDecompositionLevels(amplitude,max_level=level)
        else: levels=0

        # get time in smallest measurement unit
        start_time = time()

        for i in range(1000):
            amplitude_denoised = denoise(
                amplitude,
                level=level,
                mother_wavelet='haar',
                technique=technique,
                threshold_selection='constant',
                levels=levels,
                threshold=0
                )
        end_time = time()

        print(f'{level} {(end_time-start_time)/1000}')