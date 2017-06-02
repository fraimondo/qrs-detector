import sys

sys.path.append('../sim/')
from EdfReader import EdfReader as Er

sys.path.append('./iir1/')
import pyiir1 as iir

import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

hpass = 8
lpass = 20
order = 3


er = Er('../data/0000.edf')
er.set_channels([3])
n_blocks = 5
samps = er.read_all_samples(n_blocks).squeeze()
samps = er.read_all_samples(n_blocks).squeeze()
sfreq = er.sampling_rate(3)
filter_freq = 2.0 * lpass / np.double(sfreq)
filter_b, filter_a = butter(order, filter_freq, 'lowpass')
iir_filter = iir.ButterworthLowPass(order, sfreq, lpass)
fdata_ff = filtfilt(filter_b, filter_a, samps)
fdata_lf = lfilter(filter_b, filter_a, samps)
fdata_iir = list(samps)
iir_filter.reset()
iir_filter.filter(fdata_iir)

plt.figure()
plt.subplot(3, 3, 1)
plt.plot(fdata_ff)
plt.title('LP: filtfilt')

plt.subplot(3, 3, 2)
plt.plot(fdata_lf)
plt.title('LP: lfilter')

plt.subplot(3, 3, 3)
plt.plot(fdata_iir)
plt.title('LP: iir')


filter_freq = 2.0 * hpass / np.double(sfreq)
filter_b, filter_a = butter(order, filter_freq, 'highpass')
iir_filter = iir.ButterworthHighPass(order, sfreq, hpass)
fdata_ff = filtfilt(filter_b, filter_a, samps)
fdata_lf = lfilter(filter_b, filter_a, samps)
fdata_iir = list(samps)
iir_filter.reset()
iir_filter.filter(fdata_iir)

plt.subplot(3, 3, 4)
plt.plot(fdata_ff)
plt.title('HP: filtfilt')

plt.subplot(3, 3, 5)
plt.plot(fdata_lf)
plt.title('HP: lfilter')

plt.subplot(3, 3, 6)
plt.plot(fdata_iir)
plt.title('HP: iir')


filter_freq = [2.0 * hpass / np.double(sfreq), 2.0 * lpass / np.double(sfreq)]
filter_b, filter_a = butter(order, filter_freq, 'bandpass')
iir_filter = iir.ButterworthBandPass(order, sfreq, hpass, lpass)
fdata_ff = filtfilt(filter_b, filter_a, samps)
fdata_lf = lfilter(filter_b, filter_a, samps)
fdata_iir = list(samps)
iir_filter.reset()
iir_filter.filter(fdata_iir)


plt.subplot(3, 3, 7)
plt.plot(fdata_ff)
plt.title('BP: filtfilt')

plt.subplot(3, 3, 8)
plt.plot(fdata_lf)
plt.title('BP: lfilter')

plt.subplot(3, 3, 9)
plt.plot(fdata_iir)
plt.title('BP: iir')


plt.figure()
plt.plot(fdata_ff, 'b')
# plt.plot(fdata_lf, 'r')
plt.plot(fdata_iir, 'g')
# plt.plot(samps, 'k')
