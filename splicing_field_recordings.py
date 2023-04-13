import os
from scipy.io import wavfile
import numpy as np
import scipy.signal

directory = '/Users/kaspar/Downloads/Sounds/Cat F. catus 2s'
directoryToSave = '/Users/kaspar/Downloads/Sounds/temp cat'

noise_path = '/Users/kaspar/Downloads/Sounds/Noise 2s'
noise_clips = []

for i, filename in enumerate(sorted(os.listdir(noise_path))):
    if not filename in ['.DS_Store']:
        pathname = os.path.join(noise_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        noise_clips.append(signal)

clean_signals_path = '/Users/kaspar/Downloads/Sounds/Manually Cleaned'
clean_clips = []

for i, filename in enumerate(sorted(os.listdir(clean_signals_path))):
    if not filename=='.DS_Store':
        pathname = os.path.join(clean_signals_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2]
        clean_clips.append(signal)

PSD_Originals = []
for original in clean_clips:
    (f,S)=scipy.signal.periodogram(original.astype(np.float64),fs,scaling='density',return_onesided=True)
    print(len(S))
    PSD_Originals.append(S)
        # newname = str(count)+'.wav'
        # if count<10:
        #     newname='0'+newname
    
        # newpath = os.path.join(directoryToSave,newname)
        # wavfile.write(newpath, fs, signal)

        # count+=1