import csv
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3,3,figsize=(10,10))

for style in ['multi','scaled','full']:

    data_out_path = 'wavelet-codebase\wavelet_results_' + style + '.csv'

    match(style):
        case 'multi':
            ax_col = 0
        case 'scaled':
            ax_col = 2
        case 'full':
            ax_col = 1

    # open data out for reading
    with open(data_out_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        print(data)
        #delete first row
        data.pop(0)
        #convert to float
        data = [[float(x) for x in row] for row in data]

    print(data[0])
    # data rows look like this: [level, intensity, snnr_mean, snnr_std, sr_mean, sr_std, sdi_mean, sdi_std]

    level = [row[0] for row in data]
    level = range(1,10)
    intensity = [row[1] for row in data]
    snnr_mean = [row[2] for row in data]
    snnr_std = [row[3] for row in data]
    sr_mean = [row[4] for row in data]
    sr_std = [row[5] for row in data]
    sdi_mean = [row[6] for row in data]
    sdi_std = [row[7] for row in data]

    for metric in ['snnr', 'sr', 'sdi']:
        match(metric):
            case 'snnr':
                ax = axs[0][ax_col]
                ax.set_ylim([1,6])
                metric_mean = snnr_mean
                metric_std = snnr_std
            case 'sr':
                ax = axs[1][ax_col]
                ax.set_ylim([0.2,1])
                metric_mean = sr_mean
                metric_std = sr_std
            case 'sdi':
                ax = axs[2][ax_col]
                ax.set_ylim([9,14])
                metric_mean = sdi_mean
                metric_std = sdi_std
        
        metric_std = [x / 8 for x in metric_std]
        # plot metric against levels for each intensity
        ax.plot(level, metric_mean[0:9], label='intensity 5', marker='x', color='#000000')
        ax.plot(level, metric_mean[9:18], label='intensity 8', marker='x', color='magenta')
        ax.plot(level, metric_mean[18:27], label='intensity 11', marker='x', color='#1ff037')

        # add error regions
        # ax.fill_between(level, [x - y for x, y in zip(metric_mean[0:9], metric_std[0:9])], [x + y for x, y in zip(metric_mean[0:9], metric_std[0:9])], alpha=0.2, color='#000000')
        # ax.fill_between(level, [x - y for x, y in zip(metric_mean[9:18], metric_std[9:18])], [x + y for x, y in zip(metric_mean[9:18], metric_std[9:18])], alpha=0.2, color='magenta')
        # ax.fill_between(level, [x - y for x, y in zip(metric_mean[18:27], metric_std[18:27])], [x + y for x, y in zip(metric_mean[18:27], metric_std[18:27])], alpha=0.2, color='#1ff037')

#set titles on top 3
axs[0][0].set_title('Multi-level',fontsize=12,fontweight='bold')
axs[0][1].set_title('Full-Decomposition',fontsize=12,fontweight='bold')
axs[0][2].set_title('Scaled-Decomposition',fontsize=12,fontweight='bold')

#set y labels on left
axs[0][0].set_ylabel('SnNR (dB)',fontsize=12)
axs[1][0].set_ylabel('SR',fontsize=12)
axs[2][0].set_ylabel('SDI',fontsize=12)

#set level label on bottom
fig.text(0.5, 0.04, 'Maximum decomposition level', ha='center', va='center',fontsize=12)

handles = [plt.Line2D([],[],color='#000000',marker='x',label='intensity 3'),plt.Line2D([],[],color='magenta',marker='x',label='intensity 5'),plt.Line2D([],[],color='#1ff037',marker='x',label='intensity 7')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,1),frameon=False,ncol=3,fontsize=14)
fig.subplots_adjust(top=0.92,bottom=0.1,left=0.07,right=0.93,hspace=0.18,wspace=0.2)
plt.show()