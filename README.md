# wavelet-codebase
a collection of functions and tests to evaluate a wavelet denoising method developed for use in bioacoustics.

All Wavelet Decomposition and Evaluation functions are contained in 'waveletfuncs.py'.
'gen_...' files are included to demonstrate uses of wavelet functions.

Datasets:
Cat F. Catus 2s: 2 second recordings containing cat vocalisations as .wav, with a 'times.txt' file included.
Possum T. vulpecula 2s: 2 second recordings containing possum vocalisations as .wav, with a 'times.txt' file included.
possum_2s_manually_cleaned: 2 second recordings of possum vocalisations as .wav, which have been manually denoised using audacity.

'times.txt' files list the start and end times of signal and noise, in seconds, in each wav file in the format:
signal_start signal_end noise_start noise_end
