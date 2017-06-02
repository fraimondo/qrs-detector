import time
import sys
import math

sys.path.append('../')
from qrs_detect import QRSDetector
from qrs_predict import QRSPredictor
from EdfReader import EdfReader as Er

import numpy as np
from scipy.signal import filtfilt

# fname = '../data/test257-280-3.edf'
fname = '../../mri/data/PHYSIENS_2014-10-100004.edf'

# secs = 10
secs = 400
channel = 2
channel = 6
buf_size = 40
beta = None
beta = 0.95
offset = 1000

er = Er(fname)
er.set_channels([channel])

samps = er.read_all_samples(secs).squeeze()[offset:]
detect_params = {'beta': beta}

n_samps = len(samps)
sfreq = er.sampling_rate(channel)
# qrsl = int(math.ceil(sfreq * 0.097))
# hbl = int(math.ceil(sfreq * 0.611))
q = QRSDetector(sfreq=sfreq, debug=True, beta=beta)
peaks = np.array([])
detect_times = np.zeros((n_samps, 1), dtype=np.double)
for i, ist in enumerate(range(0, n_samps, buf_size)):
    st = time.time()
    new_peaks = q.add_detect(samps[ist:ist + buf_size])
    if len(new_peaks) > 0:
        peaks = np.append(peaks, new_peaks)
    end = time.time()
    detect_times[i] = end - st

w = np.ones((3))
osamps = np.copy(samps)
predictor = QRSPredictor(sfreq=sfreq, weights=w, detect_params=detect_params)
predictions = np.array([])
predictions_samps = np.array([])
pred_peaks = np.array([])
pred_missed = np.array([])
predict_times = np.zeros((n_samps, 1), dtype=np.double)
for i, ist in enumerate(range(0, n_samps, buf_size)):
    st = time.time()
    predictor.add_predict(samps[ist:ist + buf_size])
    p, m = predictor.get_peaks()
    peak_added = False
    if len(p) > 0 or len(m) > 0:
        peak_added = True
        print predictor.sample, '=> New Peaks', p, 'New Missed', m
    pred_peaks = np.append(pred_peaks, p)
    pred_missed = np.append(pred_missed, m)
    prediction = predictor.next()
    if prediction is not None:
        if prediction not in predictions:
            # if overrides and len(predictions) > 0:
            #     last_prediction = predictions[-1]
            #     print predictor.sample, '=> Override Prediction', last_prediction, 'with', prediction
            #     predictions[-1] = prediction
            #     predictions_samps[-1] = predictor.sample
            # else:
            text = 'FAILED' if prediction <= predictor.sample else ''
            print predictor.sample, '=> New Prediction', prediction, text
            if prediction <= predictor.sample:
                # I should debug this... it should never happen
                import pdb; pdb.set_trace()
            predictions = np.append(predictions, prediction)
            predictions_samps = np.append(predictions_samps, predictor.sample)
    end = time.time()
    predict_times[i] = end - st

import matplotlib.pyplot as plt

fsamps = filtfilt(q.filter_b_, q.filter_a_, np.copy(osamps))
plt.figure()
plt.plot(fsamps)

[plt.axvline(x, color='r', ls='--') for x in peaks]
[plt.axvline(x, color='orange', ls='--') for x in predictions]
[plt.axvline(x, color='g', ls='--') for x in predictor.missed_peaks]

plt.legend(['Raw signal', 'IIR Realtime', 'Predicted'])
plt.show()
