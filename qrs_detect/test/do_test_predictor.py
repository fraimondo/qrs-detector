import time
import sys
import math

sys.path.append('../sim/')
from qrs_detect import QRSDetector, QRSPredictor
from EdfReader import EdfReader as Er

import numpy as np

fname = '../data/test257-280-3.edf'
# fname = '../data/PHYSIENS_2014-10-100004.edf'

secs = 10
# secs = 40
channel = 2
# channel = 6
buf_size = 2


er = Er(fname)
er.set_channels([channel])
buf_size = 10

samps = er.read_all_samples(secs).squeeze()

n_samps = len(samps)
sfreq = er.sampling_rate(2)
qrsl = int(math.ceil(sfreq * 0.097))
hbl = int(math.ceil(sfreq * 0.611))
q = QRSDetector(sfreq=sfreq, debug=True)
peaks = []
detect_times = np.zeros((n_samps, 1), dtype=np.double)
for i, ist in enumerate(range(0, n_samps, buf_size)):
    st = time.time()
    new_peaks, n_samp = q.add_detect(samps[ist:ist + buf_size])
    if len(new_peaks) > 0:
        peaks = peaks + new_peaks.tolist()
    end = time.time()
    detect_times[i] = end - st

print peaks
peaks = np.array(peaks).ravel()
all_predictions = []
all_predictions_samps = []

# weights = [
#     (np.array([1]), 'constant n=1'),
#     (np.ones((3)), 'constant n=3'),
#     (np.ones((5)), 'constant n=5'),
#     (np.arange(1, 3), 'linear n=3'),
#     (np.arange(1, 6), 'linear n=5'),
#     (np.arange(1, 11), 'linear n=10')
# ]
#
import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 3)
#
#
# plt.title('Diferences Prediction - Peaks')
# for i, (ax, (w, label)) in enumerate(zip(axes.ravel(), weights)):
#
#     q = QRSPredictor(sfreq=sfreq, weights=w)
#     predictions = []
#     predictions_samps = []
#     predict_times = np.zeros((n_samps, 1), dtype=np.double)
#     for i, ist in enumerate(range(0, n_samps, buf_size)):
#         st = time.time()
#         p, d, m, samp = q.predict(samps[ist:ist + buf_size])
#         if d is not None and len(d) > 0:
#             print d
#         if p is not None:
#             if p not in predictions:
#                 predictions.append(p)
#                 predictions_samps.append(q.detector.nsamples)
#         end = time.time()
#         predict_times[i] = end - st
#
#     predictions = np.array(predictions).ravel()

# predictions_samps = np.array(predictions_samps).ravel()
# all_predictions.append(predictions)
# all_predictions_samps.append(predictions_samps)
# dif = peaks[2:] - predictions[:-1]
# ax.hist(dif, 50)
# ax.set_title(label)
# m = np.median(dif)
# ax.axvline(m, color='r', label='median {:.2f}'.format(m))
# m = np.mean(dif)
# ax.axvline(m, color='g', label='mean {:.2f}'.format(m))
# m = np.std(dif)
# ax.axvline(m, color='b', label='std {:.2f}'.format(m))
# ax.legend()
#     plt.figure()
#     plt.plot(samps)
#
#     [plt.axvline(x, color='r', ls='--') for x in peaks]
#     [plt.axvline(x, color='g', ls='--') for x in predictions]
#
#     plt.legend(['Raw signal', 'IIR Realtime', 'Predicted'])
# plt.show()
#
plt.figure()
sign = plt.plot(samps)

a = [plt.axvline(x, color='r', ls='--') for x in peaks]
b = [plt.axvline(x, color='g', ls='--') for x in predictions]

plt.legend(['Raw signal', 'IIR Realtime', 'Predicted'], [sign, a[0], b[0]])
#
# dif = peaks[2:] - predictions[:-1]
# plt.figure()
# plt.hist(dif, 50)
# plt.title('Peaks - predictions')
#
#
# plt.figure()
# dif = predictions - predictions_samps
# plt.hist(dif, 50)
# plt.title('Predictions - sample when predicting')
plt.show()
