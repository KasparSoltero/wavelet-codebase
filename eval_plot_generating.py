import matplotlib.pyplot as plt

d_levels = [0,1,2,3,4,5,6,7,8,9]
d_I_0_5 = [-11.95138583,-12.99361985,-13.08810475,-13.08424835,-13.35854964,-13.34767233,-13.3657598,-13.36876147,-13.29998352,-13.13629696]
d_I_0_8 = [-9.420714657,-11.64691651,-12.77231091,-13.09223862,-13.7195114,-13.80551325,-13.96369933,-14.11775162,-13.99942958,-14.02366773]
d_I_3_8 = [-7.690555559,-9.98621511,-11.33121565,-11.94723092,-13.13142238,-13.53715248,-13.80249552,-14.09394712,-14.23563883,-14.5586274]

d_Is = [
    d_I_0_5,
    d_I_0_8,
    d_I_3_8
]

fig = plt.figure()
fig.set_size_inches(5,3)
ax = fig.add_subplot(111)
ax.set_xlabel('Maximum Decomposition Level (No Units)')
ax.set_ylabel('SDI (No Units)')

ls = '-'
colors = ['black','magenta','green']
labels = ['0-5','0-8','3-8']

for i,d in enumerate(d_Is):
    ax.plot(d_levels,d,linestyle='--',color=colors[i],label=labels[i])

ax.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0., title='Intensity range')

plt.subplots_adjust(left=0.15,bottom=0.15)
plt.show()