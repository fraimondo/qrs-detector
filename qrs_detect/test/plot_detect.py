import sys
import math

sys.path.append('../sim/')
from qrs_detect import QRSDetector
from EdfReader import EdfReader as Er

import numpy as np
from scipy.signal import filtfilt


er = Er('../data/test257-280-4.edf')
er.set_channels([2])
n_blocks = 10
n_samps = 4000
samps = er.read_all_samples(n_blocks).squeeze()

srate = er.sampling_rate(2)
qrsl = int(math.ceil(srate * 0.097))
hbl = int(math.ceil(srate * 0.611))
q = QRSDetector(8, 20, qrsl, hbl, 0.08, sfreq=srate, debug=True)
q.append_samples(samps[:q.skip_before])
peaks = []
for ist in range(q.skip_before, n_samps):
    peaks = peaks + q.add_detect(samps[ist:ist + 1])

q.filter.reset()
r2, b2 = q.qrs_detect_iir(samps[:n_samps])
a = list(samps[:n_samps])
q.filter.reset()
q.filter.filter(a)
fsamps = a

r3, b3 = q.qrs_detect_filtfilt(samps[:n_samps])
fsamps2 = filtfilt(q.filter_b_, q.filter_a_, np.copy(samps[:n_samps]))

import matplotlib.pyplot as plt
# plt.figure()
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(samps[:n_samps])
ax1.plot(fsamps, color='g')
[ax1.axvline(x, color='r', ls='--') for x in peaks]
[ax1.axvline(x, color='r', ls=':') for x in np.array(q.blocks).ravel()]
ax1.set_title('IIR Realtime')
ax2.plot(samps[:n_samps])
ax2.plot(fsamps, color='g')
[ax2.axvline(x, color='r', ls='--') for x in r2]
[ax2.axvline(x, color='r', ls=':') for x in b2.ravel()]
ax2.set_title('IIR Forward Filter')
ax3.plot(samps[:n_samps])
ax3.plot(fsamps2, color='g')
[ax3.axvline(x, color='r', ls='--') for x in r3]
[ax3.axvline(x, color='r', ls=':') for x in b3.ravel()]
ax3.set_title('FILTFILT Filter')

plt.figure()
for i, p, c, l in zip(range(3),
                      (peaks, r2, r3),
                      ('r', 'g', 'b'),
                      ('IIR Realtime', 'IIR Forward', 'FILTFILT')):
    plt.plot(p, [1 + (i * 0.01)] * len(p), '.', color=c, label=l)

data_range = np.abs(np.min(samps[:n_samps])) + np.abs(np.max(samps[:n_samps]))
sdata = (samps[:n_samps] + np.abs(np.min(samps[:n_samps]))) / data_range
plt.plot(sdata)

plt.ylim(0.8, 1.05)
plt.legend()
plt.show()
