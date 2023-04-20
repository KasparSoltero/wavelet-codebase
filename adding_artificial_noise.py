import os
from scipy.io import wavfile
import numpy as np
import scipy.signal
import random
from waveletfuncs import denoise, SDI
from waveletfuncs import *
from graphing import *

## get noise
noise_path = '/Users/kaspar/Downloads/Sounds/Noise 2s'
noise_clips = []

for i, filename in enumerate(sorted(os.listdir(noise_path))):
    if not filename in ['.DS_Store']:
        pathname = os.path.join(noise_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        noise_clips.append(signal.astype(object))

## add noise to clean signals randomly
clean_signals_path = '/Users/kaspar/Downloads/Sounds/Manually Cleaned'
clean_clips = []
dirty_clips = []

for i, filename in enumerate(sorted(os.listdir(clean_signals_path))):
    if not filename=='.DS_Store':
        pathname = os.path.join(clean_signals_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        clean_clips.append(signal.astype(object))
        
        random_noise = random.choice(noise_clips)
        dirty_clips.append(np.add(signal.astype(object),random_noise.astype(object)))
        # print(f'{filename} {len(signal)}')

## save clips which have artificial noise added
for i,sig in enumerate(dirty_clips):
    name = str(i)+'.wav'
    if i<10:
        name = '0'+name
    name = '/Users/kaspar/Downloads/sounds/Artificial Noise/'+name
    wavfile.write(name,96000,sig.astype(np.int16))
