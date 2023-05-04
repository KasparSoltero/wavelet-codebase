import csv
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3,1,figsize=(5,9))

time_d_method_data = 'wavelet-codebase\wavelet_results_' + 'time_d_method' + '.csv'
# open data out for reading
with open(time_d_method_data, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    #delete first row
    data.pop(0)
    #convert to float
    print(data)
    data = [[float(x) for x in row] for row in data if row[0]]
level = [row[0] for row in data]
multi_res = [row[1] for row in data]
full_res = [row[2] for row in data]
scaled_possum = [row[3] for row in data]
scaled_cat = [row[4] for row in data]
scaled_bird = [row[5] for row in data]


for style in ['multi','scaled','full']:

    data_out_path = 'wavelet-codebase\wavelet_results_' + style + '.csv'

    match(style):
        case 'multi':
            times=multi_res
            color='#000000'
        case 'scaled':
            times=scaled_cat
            color='magenta'
        case 'full':
            times=full_res
            color='#1ff037'

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

    times = times[:len(level)]

    for metric in ['snnr', 'sr', 'sdi']:
        match(metric):
            case 'snnr':
                ax = axs[0]
                ax.set_ylim([1,6])
                metric_mean = snnr_mean
                metric_std = snnr_std
            case 'sr':
                ax = axs[1]
                ax.set_ylim([0.2,1])
                metric_mean = sr_mean
                metric_std = sr_std
            case 'sdi':
                ax = axs[2]
                ax.set_ylim([8,15])
                metric_mean = sdi_mean
                metric_std = sdi_std
        
        metric_std = [x / 8 for x in metric_std]
        # plot metric against levels for each intensity
        # ax.plot(times, metric_mean[0:9], label='intensity 5', marker='x', color='#000000')
        # ax.plot(times, metric_mean[9:18], label='intensity 8', marker='x', color='magenta')
        ax.plot(times, metric_mean[18:27], label='intensity 11', marker='x', color=color)

        # add error regions
        # ax.fill_between(times, [x - y for x, y in zip(metric_mean[0:9], metric_std[0:9])], [x + y for x, y in zip(metric_mean[0:9], metric_std[0:9])], alpha=0.2, color='#000000')
        # ax.fill_between(times, [x - y for x, y in zip(metric_mean[9:18], metric_std[9:18])], [x + y for x, y in zip(metric_mean[9:18], metric_std[9:18])], alpha=0.2, color='magenta')
        ax.fill_between(times, [x - y for x, y in zip(metric_mean[18:27], metric_std[18:27])], [x + y for x, y in zip(metric_mean[18:27], metric_std[18:27])], alpha=0.2, color=color)

#set titles on top 3
# axs[0].set_title('Multi-level',fontsize=12,fontweight='bold')
# axs[0].set_title('Full-decomposition',fontsize=12,fontweight='bold')
# axs[0].set_title('Scaled-decomposition',fontsize=12,fontweight='bold')
fig.text(0.95, 0.95, 'Multi-level', ha='center', va='center',fontsize=12,rotation=90)
fig.text(0.95, 0.65, 'Full-decomposition', ha='center', va='center',fontsize=12,rotation=90)
fig.text(0.95, 0.35, 'Scaled-decomposition', ha='center', va='center',fontsize=12,rotation=90)

#set y labels on left
axs[0].set_ylabel('SnNR (dB)',fontsize=12)
axs[1].set_ylabel('SR',fontsize=12)
axs[2].set_ylabel('SDI',fontsize=12)

#set level label on bottom
fig.text(0.5, 0.04, 'Maximum decomposition level', ha='center', va='center',fontsize=12)

handles = [plt.Line2D([],[],color='#000000',marker='x',label='Multi-res'),plt.Line2D([],[],color='magenta',marker='x',label='Scaled-res'),plt.Line2D([],[],color='#1ff037',marker='x',label='Full-res')]
fig.legend(handles=handles,loc='center right',bbox_to_anchor=(0.5,1),frameon=False,ncol=3,fontsize=12)
fig.subplots_adjust(top=0.92,bottom=0.1,left=0.07,right=0.93,hspace=0.18,wspace=0.2)
plt.show()