import csv
import matplotlib.pyplot as plt
import numpy as np
from time import time


fig, axs = plt.subplots(1,1,figsize=(5,3))

ax = axs

#colors:
#'magenta'
#1ff0f0 - cyan
#1ff037 - green
#fc5017 - orange
#0836bf - dark blue

for data in ['time_d_method']:

    data_out_path = 'wavelet_results_' + data + '.csv'

    # open data out for reading
    with open(data_out_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        #delete first row
        data.pop(0)
        #convert to float
        print(data)
        data = [[float(x) for x in row] for row in data if row[0]]

    print(data[0])
    # data rows look like this: [level, multi_res, full_res, scaled_possum, scaled_cat, scaled_bird]

    level = [row[0] for row in data]
    print(level)
    # level = range(1,10)
    multi_res = [row[1] for row in data]
    full_res = [row[2] for row in data]
    scaled_possum = [row[3] for row in data]
    scaled_cat = [row[4] for row in data]
    scaled_bird = [row[5] for row in data]

    # plot metric against levels for each intensity
    ax.plot(level, multi_res, marker='', color='#000000')
    ax.plot(level, full_res, marker='', color='#0836bf')
    ax.plot(level, scaled_possum, marker='', color='#1ff037')
    ax.plot(level, scaled_cat, marker='', color='magenta')
    # ax.plot(level, scaled_bird, marker='', color='#fc5017')
#set titles on top 3

#set y labels on left
ax.set_ylabel('Mean computation time (ms)',fontsize=12)
ax.set_xlabel('Maximum decomposition depth',fontsize=12)
fig.legend(['Multi-res','Full-res','Scaled-res (possum)','Scaled-res (cat)','Scaled-res (bird)'],loc='upper left',bbox_to_anchor=(0.15,1),frameon=False,ncol=1,fontsize=12)
# handles = [plt.Line2D([],[],color='#000000',marker='x',label='intensity 3'),plt.Line2D([],[],color='magenta',marker='x',label='intensity 5'),plt.Line2D([],[],color='#1ff037',marker='x',label='intensity 7')]
# fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,1),frameon=False,ncol=3,fontsize=14)
fig.subplots_adjust(top=1,bottom=0.17,left=0.133,right=0.976)
plt.show()