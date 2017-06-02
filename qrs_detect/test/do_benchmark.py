import time
import sys
import math

sys.path.append('../sim/')
from qrs_detect import QRSDetector
from EdfReader import EdfReader as Er

import numpy as np
from scipy.signal import filtfilt


er = Er('../data/test257-280-3.edf')
er.set_channels([2])

samps = er.read_all_samples().squeeze()

n_samps = len(samps)
sfreq = er.sampling_rate(2)
qrsl = int(math.ceil(sfreq * 0.097))
hbl = int(math.ceil(sfreq * 0.611))
q = QRSDetector(8, 20, qrsl, hbl, 0.08, sfreq=sfreq, debug=True)
q.append_samples(samps[:q.skip_before])
peaks = []
times = np.zeros((n_samps - q.skip_before, 1), dtype=np.double)
for i, ist in enumerate(range(q.skip_before, n_samps)):
    st = time.time()
    peaks = peaks + q.add_detect(samps[ist:ist + 1])
    end = time.time()
    times[i] = end - st

q.filter.reset()
r2, b2 = q.qrs_detect_iir(samps)
a = list(samps)
q.filter.reset()
q.filter.filter(a)
fsamps = a

r3, b3 = q.qrs_detect_filtfilt(samps)
fsamps2 = filtfilt(q.filter_b_, q.filter_a_, np.copy(samps))

print ' Stats '
print '======='
print ' Mean time per sample (ms):', np.mean(times) * 1000
print ' Max time per sample (ms):', np.max(times) * 1000
print ' Min time per sample (ms):', np.min(times) * 1000
print ' Events detected:'
print '     {} (IIR Realtime)'.format(len(peaks))
print '     {} (IIR Forward)'.format(len(r2))
print '     {} (Filtfilt)'.format(len(r3))
print ' Mismatches:'
mismatches = np.array([], dtype=np.int)
n = np.abs(np.array(peaks) - np.array(r2))
m = np.nonzero(n)[0]
mismatches = np.concatenate((mismatches, m))
print '     {} ({:.2%}) max {} min {} mean {:.2f} IIR Realtime vs IIR Forward'\
    .format(
        len(m), float(len(m)) / len(n), np.max(m), np.min(m), np.mean(m)
    )
n = np.abs(np.array(peaks) - np.array(r3))
m = np.nonzero(n)[0]
mismatches = np.concatenate((mismatches, m))
print '     {} ({:.2%}) max {} min {} mean {:.2f} IIR Realtime vs Filtfilt'\
    .format(
        len(m), float(len(m)) / len(n), np.max(m), np.min(m), np.mean(m)
    )
n = np.abs(np.array(r2) - np.array(r3))
m = np.nonzero(n)[0]
mismatches = np.concatenate((mismatches, m))
print '     {} ({:.2%}) max {} min {} mean {:.2f} IIR Forward vs Filtfilt'\
    .format(
        len(m), float(len(m)) / len(n), np.max(m), np.min(m), np.mean(m)
    )

import matplotlib.pyplot as plt
plt.figure()
plt.plot(samps)
i = True
for x in mismatches:
    plt.axvline(peaks[x], color='r', ls='--')
    plt.axvline(r2[x], color='b', ls='--')
    plt.axvline(r3[x], color='g', ls='--')

plt.legend(['Raw signal', 'IIR Realtime', 'IIR Forward', 'Filtfilt'])
plt.show()
