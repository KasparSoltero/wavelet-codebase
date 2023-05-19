import os
from scipy.io import wavfile
from graphing import *

# Set paths
sounds_path = './sounds'
possum_path = os.path.join(sounds_path,'Possum T. vulpecula 2s')
cat_path = os.path.join(sounds_path,'Cat F. Catus 2s')
munually_cleaned_path = os.path.join(sounds_path,'Manually Cleaned')
artificial_noise_added_path = os.path.join(sounds_path,'Artificial Noise')
data_out_path = './wavelet_results.xlsx'

# define which animal sounds to use
sound_file_path = possum_path

### Opens selected files in order 00 - 01
for i, filename in enumerate(sorted(os.listdir(sound_file_path))):
    ### which files to see:
    # if not filename in ['.DS_Store', 'times.txt']:
    # if filename in high_SNR_files_possum:
    if filename=='34.wav':

        ### load file & data
        pathname = os.path.join(sound_file_path,filename)
        [fs, signal] = wavfile.read(pathname)
        signal = signal[:fs*2] #make sure all files are 2 seconds exactly
        # print('{:<20},{:<20}'.format(filename,len(signal)))
        signal=signal.astype(object) #allows numpy to be more precise?

        #####Choose internal loop
        frequency_decomposition_visual(signal)

plt.show()