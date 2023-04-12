import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy as np
from math import floor
from waveletfuncs import getScaledDecompositionLevels, getHurstThresholds, getFilters, scaledres, getEntropyThresholds,getStdThresholds, normaliseE, normaliseH,normaliseS

def plot_specgram(data, fs=96000, title='', x_label='', y_label='', fig_size=[5,3],cmap='inferno'):
    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # vmin = 20*np.log10(np.max(data)) - 100
    vmax = 40
    vmin = -30
    pxx,  freq, t, cax = plt.specgram(data, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
    fig.colorbar(cax).set_label('Intensity [dB]')

def add_specgram(data,ax,fs=96000,cmap='inferno'):
    vmax = 40
    vmin = -30
    cax=ax.specgram(data,Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
    # ax.colorbar(cax).set_label('Intensity [dB]')
    return cax

def plot_histogram(coeffs,bins=30,title='',x_label='bins',y_label='count',fig_size=[5,3]):
    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    colors = ['black','green','red','pink','blue','yellow','grey','brown']
    for i in range(len(coeffs)):
        data = np.abs(coeffs[i])
        n, bins, patches = plt.hist(data, bins=bins, density=True, histtype='step')
        # y = ((1 / (np.sqrt(2 * np.pi) * np.std(data))) * np.exp(-0.5 * (1 / np.std(data) * (bins - np.mean(data)))**2))
        plt.stairs(n,bins,linewidth='4',color=colors[i])
    plt.legend([str(i) for i in range(len(coeffs))])

def add_signal_noise_patch(ax,times,fs=96000,sig_color='#1ff037',noise_color='#cccccc'):
    [sigstart,sigend,noisestart,noiseend] = times

    linestyle='--'
    lw = 2
    minleft = 0.005
    minbottom = 500
    minseparate_x = 0.01
    height = (fs/2) - 2*minbottom

    ## convert to seconds
    sigstart = sigstart
    sigend = sigend
    noisestart = noisestart
    noiseend = noiseend

    ## signal patch
    if sigstart<=minleft:
        sigstart=minleft
    elif ((sigstart-noiseend) >= 0) and ((sigstart-noiseend)<=minseparate_x):
        sigstart = noiseend + minseparate_x
    width = sigend-sigstart
    ax.add_patch(Rectangle((sigstart,minbottom), width, height, edgecolor=sig_color, linestyle=linestyle, fill=False, lw=lw))

    ## noise patch
    if noisestart<=minleft:
        noisestart=minleft
    elif ((noisestart-sigend) >= 0) and ((noisestart-sigend)<=minseparate_x):
        noisestart = sigend + minseparate_x
    width = noiseend-noisestart
    ax.add_patch(Rectangle((noisestart,minbottom), width, height, edgecolor=noise_color, linestyle=linestyle, fill=False, lw=lw))

def four_specgrams(data,fig_size=[5,3]):
    fs = 96000
    color = 'magenta'
    linewidth = 1.5
    cmap = 'Greys'

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(fig_size[0], fig_size[1])

    add_specgram(data,axs[0,0],cmap=cmap)
    axs[0,0].set_xlim(0,2)
    axs[0,0].get_xaxis().set_visible(False)
    freqs = [fs/4]
    axs[0,0].hlines(freqs,0,2,linewidth=linewidth,colors=[color]*len(freqs))
    axs[0,0].title.set_text('a)')

    add_specgram(data,axs[0,1],cmap=cmap)
    axs[0,1].set_xlim(0,2)
    axs[0,1].get_xaxis().set_visible(False)
    axs[0,1].get_yaxis().set_visible(False)
    freqs = [fs/4,fs/8,fs/16]
    axs[0,1].hlines(freqs,0,2,linewidth=linewidth,colors=[color]*len(freqs))
    axs[0,1].title.set_text('b)')

    add_specgram(data,axs[1,0],cmap=cmap)
    axs[1,0].set_xlim(0,2)
    freqs = [i*fs/16 for i in range(1,8)]
    axs[1,0].hlines(freqs,0,2,linewidth=linewidth,colors=[color]*len(freqs))
    axs[1,0].title.set_text('c)')

    cax = add_specgram(data,axs[1,1],cmap=cmap)
    axs[1,1].set_xlim(0,2)
    axs[1,1].get_yaxis().set_visible(False)
    freqs = [i*fs/16 for i in [1,2,3,4,6]]
    axs[1,1].hlines(freqs,0,2,linewidth=linewidth,colors=[color]*len(freqs))
    axs[1,1].title.set_text('d)')

    plt.subplots_adjust(
        left=0.19,
        bottom=0.16,
        wspace=0.15,
        hspace=0.3
    )
    fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical')
    fig.text(0.5,0.04,'Time (s)',ha='center')
    fig.colorbar(cax[3],ax=axs.ravel().tolist())
    # fig.subplots_adjust(left=0.16,bottom=0.12)

def plot_hurst(data,fig_size=[5,3],title='',x_label='',y_label=''):
    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    hurst_vals = data[0]
    hurst_indices = data[1]

    plt.plot(hurst_indices,hurst_vals)

    # pxx,  freq, t, cax = plt.specgram(data, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
    # fig.colorbar(cax).set_label('Intensity [dB]')

def scaled_bands_visual(signal,level=5,windows=5,fs=96000,title='',x_label='',y_label='',fig_size=[5,3],cmap='inferno',levels=''):
    # grid type visual with colors by hurst values

    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # vmin = 20*np.log10(np.max(data)) - 100

    vmax = 40
    vmin = -30
    pxx,  freq, t, cax = plt.specgram(signal, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
    fig.colorbar(cax).set_label('Intensity [dB]')

    linestyle='--'
    edgecolor='#fff'
    lw = 1 #line width
    windows_count = windows

    [lpf_R,lpf_D,hpf_R,hpf_D] = getFilters('haar')
    if not levels:
        levels,level_depths = getScaledDecompositionLevels(signal,max_level=level)    
    coeffs,lengthsArray = scaledres(signal,levels,lpf_D,hpf_D)
    min_bandwidth = int((fs/2)/(2**level))
    print(f'min bw {min_bandwidth}')
    H_arrays = []
    horizontal_indices_seconds = []
    for frequency_band in coeffs:
        thres_array, H_indices, H_array = getHurstThresholds(frequency_band,max_windows=windows_count)
        H_arrays.append(H_array)
        horizontal_indices_seconds.append(2*np.divide(H_indices,len(frequency_band)))

    frequency_counter = 0
    freq_to_draw = 0
    i=0
    for val in level_depths:
        if frequency_counter>= freq_to_draw:
            bw = min_bandwidth*2**(level-val)
            # ax.add_patch(Rectangle((0,freq_to_draw), 2, bw, edgecolor=edgecolor, linestyle=linestyle, fill=False, lw=lw))
            ax.plot([0,2],[freq_to_draw,freq_to_draw],color=edgecolor,linestyle=linestyle,lw=lw)
            freq_to_draw += bw
            i+=1
        frequency_counter += min_bandwidth
    
    # ax.set_xlim([0,2])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    plt.subplots_adjust(bottom=0.15, left=0.15)

def noise_grid_visual(signal,method='hurst',level=5,windows=5,fs=96000,title='',x_label='',y_label='',fig_size=[5,3],cmap='inferno',levels=''):
    # grid type visual with colors by hurst values

    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_size, gridspec_kw={'width_ratios': [25,1]})    
    fig.set_size_inches(fig_size[0], fig_size[1])
    # ax = fig.add_subplot(121)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # vmin = 20*np.log10(np.max(data)) - 100

    vmax = 40
    vmin = -30
    pxx,  freq, t, cax = ax.specgram(signal, Fs=fs,cmap=cmap,vmin=vmin,vmax=vmax)
    # fig.colorbar(cax).set_label('Intensity [dB]')

    edgecolor='#fff'
    lw = 0 #line width
    windows_count = windows

    [lpf_R,lpf_D,hpf_R,hpf_D] = getFilters('haar')
    if not levels:
        levels,level_depths = getScaledDecompositionLevels(signal,max_level=level)    
    coeffs,lengthsArray = scaledres(signal,levels,lpf_D,hpf_D)
    min_bandwidth = int((fs/2)/(2**level))
    print(f'min bw {min_bandwidth}')

    globalMedian = np.median([abs(item) for sublist in coeffs for item in sublist])
    globalStd = np.std([abs(item) for sublist in coeffs for item in sublist])

    H_arrays = []
    horizontal_indices_seconds = []
    for frequency_band in coeffs:
        if method=='hurst':
            thres_array, H_indices, H_array = getHurstThresholds(frequency_band,max_windows=windows_count)
        elif method=='entropy':
            thres_array, H_indices, H_array = getEntropyThresholds(frequency_band,max_windows=windows_count)
        elif method=='std':
            thres_array, H_indices, H_array = getStdThresholds(frequency_band,max_windows=windows_count,globalMedian=globalMedian,globalStd=globalStd)
        H_arrays.append(H_array)
        horizontal_indices_seconds.append(2*np.divide(H_indices,len(frequency_band)))

    minmin = 9999999
    maxmax = -999999
    for arr in H_arrays:
        if min(arr)<minmin: minmin = min(arr)
        if max(arr)>maxmax: maxmax = max(arr)
    print(f'min and max was {minmin}, {maxmax}, calibrating max: {globalStd*4}')

    frequency_counter = 0
    freq_to_draw = 0
    i=0
    for val in level_depths:
        if frequency_counter>= freq_to_draw:
            bw = min_bandwidth*2**(level-val)
            # print(f'i: {i} val {val} for level {level} bw {bw} hurst values {H_arrays[i]} indices {horizontal_indices_seconds[i]}')
            for j in range(len(horizontal_indices_seconds[i])):
                x = horizontal_indices_seconds[i][j]
                h_val = H_arrays[i][j]
                if method=='hurst':
                    h_val = abs(normaliseH(h_val))
                elif method=='entropy':
                    h_val = normaliseE(h_val)
                elif method=='std':
                    h_val = normaliseS(h_val,globalStd)
                hurst_color = (h_val,h_val,h_val,1)
                ax.add_patch(Rectangle((x,freq_to_draw), 2, bw, color=hurst_color, fill=True, lw=lw))
            freq_to_draw += bw
            i+=1
        frequency_counter += min_bandwidth
        
    ax.set_xlim([0,2])

    # greyscale colorbar
    # bar_ax = fig.add_subplot(122)
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=50, vmax=globalStd*4)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax2, orientation='vertical', label='Standard Deviation of Ỹ [dB]')
    
    plt.subplots_adjust(bottom=0.15, left=0.15, right=0.82)
