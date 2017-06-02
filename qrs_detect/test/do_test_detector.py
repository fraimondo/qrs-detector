import time
import sys
import math

sys.path.append('../sim/')
from qrs_detect import QRSDetector
from EdfReader import EdfReader as Er

import numpy as np
from scipy.signal import filtfilt


fname = '../data/PHYSIENS_2014-10-100004.edf'
secs = 40
channel = 6
buf_size = 2

er = Er(fname)
er.set_channels([channel])
sfreq = er.sampling_rate(channel)

n_samps = 40 * sfreq
samps = er.read_all_samples(secs).squeeze()

from scipy import io as sio
mc = sio.loadmat('data1k.mat')
samps = np.squeeze(mc['data'].T)
sfreq = 83
osamps = np.copy(samps)

n_samps = len(samps)
# qrsl = int(math.ceil(sfreq * 0.097))
# hbl = int(math.ceil(sfreq * 0.611))
q = QRSDetector(sfreq=sfreq, debug=True, beta=0.9)
peaks = []
detect_times = np.zeros((n_samps, 1), dtype=np.double)
for i, ist in enumerate(range(0, n_samps, buf_size)):
    st = time.time()
    new_peaks = q.add_detect(samps[ist:ist + buf_size])
    if len(new_peaks) > 0:
        print new_peaks
    peaks = np.append(peaks, new_peaks)
    end = time.time()
    detect_times[i] = end - st

peaks = np.array(peaks).ravel()


import matplotlib.pyplot as plt

fsamps = filtfilt(q.filter_b_, q.filter_a_, np.copy(osamps))

plt.figure()
plt.plot(fsamps)
[plt.axvline(x, color='r', ls='--') for x in peaks]
plt.show()
