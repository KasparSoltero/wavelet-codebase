from waveletfuncs import *
from math import sin, pi
import numpy as np
import matplotlib.pyplot as plt

fs = 100
t = np.arange(0,1,1/fs)
f1 = 4
f2 = 0

S = 5*np.sin(2*pi*f1*t) + np.sin(2*pi*f2*t)
N = np.random.normal(0,1,len(S))
S = S + N

level = 1
fig, axs = plt.subplots(level+2,1,figsize=(5,9))

axs[0].plot(S)

[lpf_D, hpf_D, lpf_R, hpf_R] = getFilters('sym20')
cA = S
coeffs = [S]

for i in range(level):
    [cA,cD] = dwt(cA,lpf_D,hpf_D)

    coeffs[len(coeffs)-1] = cD
    coeffs.append([0]*len(cA))

    print(coeffs)

    reconstructed = inverseMultires(S,coeffs, lpf_R, hpf_R)
    axs[i+1].plot(reconstructed)
    coeffs[len(coeffs)-2] = [0]*len(cD)

coeffs[len(coeffs)-1] = cA
reconstructed = inverseMultires(S,coeffs, lpf_R, hpf_R)
print(coeffs)
axs[level+1].plot(reconstructed)

# axs[0].plot(S)
# axs[1].plot(idwt(cD*0,cD,len(S),lpf_R,hpf_R))
# axs[3].plot(idwt(cA*0,cA,len(S),lpf_R,hpf_R))


plt.show()