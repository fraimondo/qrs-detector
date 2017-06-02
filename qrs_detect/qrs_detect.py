from __future__ import print_function
import sys
sys.path.append('./iir1/')
use_pyiir = False
try:
    from .iir import pyiir1 as iir
    use_pyiir = True
    print('Using Realtime IIR filter')
except:
    print ('Not using Realtime IIR filter')
import math
import numpy as np
from scipy.signal import butter, filtfilt


class RingBuffer(object):
    def __init__(self, max_size, rows=1, dtype=np.float):
        self.size = 0
        self.max_size = max_size
        self.buffer = np.zeros((self.max_size, rows), dtype=dtype)
        self.counter = 0

    def append(self, data, auto=False):
        """this is an O(n) operation"""
        n = len(data)
        # if auto is True:
        #   print self.remaining, self.size, '->',
        if self.max_size - len(self) < n:
            if auto is True:
                # print 'Auto consume', n - (self.max_size - len(self)), '->',
                self.consume(n - (self.max_size - len(self)))
            else:
                raise RuntimeError("Buffer Overflow")
        if self.remaining < n:
            # print 'compacting',
            self.compact()
        # if auto is True:
            # print self.remaining, self.size
        if isinstance(data, np.ndarray):
            if len(data.shape) < 2:
                self.buffer[self.counter + self.size:][:n] = data[:, None]
            else:
                self.buffer[self.counter + self.size:][:n] = data
        else:
            self.buffer[self.counter + self.size:][:n] = np.array(data)[:, None]

        self.size += n

    def consume(self, n):
        self.counter += n
        self.size -= n

    @property
    def remaining(self):
        return self.max_size - (self.counter + self.size)

    def compact(self):
        """
        note: only when this function is called, is an O(size)
        performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        self.buffer[:self.size] = self.view
        self.counter = 0

    def ravel(self):
        return self.view.ravel()

    @property
    def view(self):
        """this is always an O(1) operation"""
        return self.buffer[self.counter:][:self.size]

    def cumsum(self, axis, dtype, out):
        return np.cumsum(self.buffer[self.counter:][:self.size],
                         axis, dtype, out)

    def mean(self, axis, dtype, out, keepdims=False):
        return np.mean(self.buffer[self.counter:][:self.size],
                       axis, dtype, out, keepdims)

    # def argmax(self, axis):
    #     return np.argmax(self.buffer[self.counter:][:self.size], axis)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
        elif isinstance(index, int):
            start = index
            stop = index + 1
        if start is None:
            start = 0
        if stop is None:
            stop = 0
        if start < 0:
            start += self.size
            stop += self.size
        if stop > self.size:
            raise RuntimeError("Slice end cannot be bigger than size")
        elif stop < 0:
            stop += self.size

        if start < 0:
            raise RuntimeError("Out of bounds")
        stop += self.counter
        start += self.counter
        return self.buffer[start:stop]

    def __repr__(self):
        return self.view.__repr__()


def moving_average(a, n):
    ret = np.cumsum(a, dtype=np.double)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class QRSDetector():
    def __init__(self, sfreq, hpass=None, lpass=None, qrs_length=None,
                 beat_length=None, beta=None, debug=False):
        if hpass is None:
            hpass = 8.0
        if lpass is None:
            lpass = 20.0
        if qrs_length is None:
            qrs_length = int(math.ceil(sfreq * 0.097))
        if beat_length is None:
            beat_length = int(math.ceil(sfreq * 0.611))
        if beta is None:
            beta = 0.08
        self.lpass = lpass
        self.hpass = hpass
        self.qrs_length = qrs_length
        self.beat_length = beat_length
        to_skip = (self.beat_length - self.qrs_length)
        self.skip_before = int(to_skip / 2 if to_skip % 2 == 0 else to_skip / 2
                               + 1)
        self.skip_after = int(to_skip / 2)
        self.beta = beta
        self.sfreq = sfreq
        self._update_filter()

        self.reset()

        self.debug = debug
        if debug is True:
            self.blocks = []
            self.peaks = []

    def reset(self):
        if use_pyiir:
            self.filter.reset()
        self.buffer = RingBuffer(1000 * self.beat_length)
        self.data = RingBuffer(1000 * self.beat_length)
        self.nsamples = 0
        self.mean = 0
        self.boi = 0

    def set_beta(self, value):
        self.beta = value

    def __repr__(self):
        """Representation"""
        # values
        s = ' n_data: %d,' % len(self.data)
        s += ' samples: %d' % self.nsamples
        e_s = 'hpass: {}, lpass: {}, beat_length: {}, qrs_length: {}'.format(
            self.hpass, self.lpass, self.beat_length, self.qrs_length,
        )
        e_s += ', skip_before: {}, skip_after: {}'.format(
            self.skip_before, self.skip_after)
        if self.debug:
            e_s += ' DEBUG MODE'
        s = '<QRS Detector %s | %s>' % (e_s, s)
        return s

    @property
    def delay(self):
        return self.beat_length - self.qrs_length

    def _update_filter(self):
        filter_freq = [2.0 * self.hpass / np.double(self.sfreq),
                       2.0 * self.lpass / np.double(self.sfreq)]
        print('Filtering at ', [self.hpass, self.lpass], 'Hz (bandpass)')
        # Butterworth Order 3 (Page 2 Fast QRS Detection from M. Elgendi)
        self.filter_b_, self.filter_a_ = butter(3, filter_freq, 'bandpass')
        if use_pyiir:
            self.filter = iir.ButterworthBandPass(
                3, self.sfreq, self.hpass, self.lpass)
            self.filter.reset()

    def qrs_detect_filtfilt(self, data):
        # Filter -> Square
        fdata = np.square(filtfilt(self.filter_b_, self.filter_a_, data))
        # skip = (self.beat_length - self.qrs_length) / 2
        ma_qrs = moving_average(fdata[self.skip_before:-self.skip_after],
                                self.qrs_length)
        ma_beat = moving_average(fdata, self.beat_length)
        z = fdata.mean()
        alpha = self.beta * z
        thr1 = ma_beat + alpha
        boi = np.append(ma_qrs > thr1, False)
        boi = np.insert(boi, 0, False)
        blocks = np.where(boi[1:] - boi[:-1])[0].reshape(-1, 2)
        thr2 = self.qrs_length
        peaks = []
        for bi in range(blocks.shape[0]):
            s = blocks[bi, 0] + self.skip_before
            e = blocks[bi, 1] + self.skip_before
            if (e - s) > thr2:
                peaks.append(s + np.argmax(np.abs(data[s:e])))
        blocks += self.skip_before

        return peaks, blocks

    def _preprocess_append(self, samps):
        if isinstance(samps, np.ndarray):
            samps = samps.tolist()
        elif not isinstance(samps, list):
            samps = [samps]
        # import pdb; pdb.set_trace()
        self.data.append(np.copy(samps))
        n = len(samps)
        self.filter.filter(samps)
        samps = np.square(samps)
        v = np.sum(samps)
        self.mean = float(self.mean * self.nsamples + v) / (self.nsamples + n)
        self.nsamples += n
        self.buffer.append(samps)

    def _consume(self, nsamps=None):
        if nsamps is None:
            nsamps = len(self.buffer) - self.beat_length
        # print 'Consume =>', nsamps
        self.buffer.consume(nsamps)
        self.data.consume(nsamps)

    def add_to_beta(self, delta):
        self.beta += delta

    def append_samples(self, samps):
        ''' Add samples to the detector, but don't detect.
            Keeps the buffer size to the minimum
        '''
        self._preprocess_append(samps)
        dif = len(self.buffer) - self.beat_length - self.skip_before
        if dif > 0:
            self._consume(dif)

    def add_detect(self, samps):
        # n_samps = len(samps)
        # print 'Detecting', n_samps
        block_start = self.nsamples - len(self.buffer) + self.skip_before
        self._preprocess_append(samps)

        if len(self.buffer) <= self.beat_length:
            return []
        self._consume(1)  # Remove the first

        # Perform the moving average over the qrs
        ma_qrs = moving_average(
            self.buffer[self.skip_before:-self.skip_after],
            self.qrs_length)

        # Perform the moving average over the beat
        ma_beat = moving_average(
            self.buffer, self.beat_length)

        # We must have the same number of averaged elements than samps added
        # print len(ma_beat), n_samps
        # assert(len(ma_beat) == n_samps)

        thr1 = self.beta * self.mean + ma_beat

        peaks = []
        new_bois = ma_qrs > thr1
        new_bois = np.insert(new_bois, 0, self.boi)

        changes = np.where(new_bois[1:] - new_bois[:-1])[0]
        if self.boi and len(changes) == 0:
            # No changes, update max BOI
            tdata = self.data[self.skip_before]
            idx = block_start + 1
            if self.boi_max < tdata:
                self.boi_max = tdata
                self.boi_max_idx = idx
        if self.boi and len(changes > 1):
            # First element in changes must be the boi end
            end = changes[0] + block_start
            if (end - self.boi_start) > self.qrs_length:
                e = changes[0] + self.skip_before
                boi_max_idx = np.argmax(self.data[self.skip_before - 1:e])
                boi_max = self.data[boi_max_idx]
                boi_max_idx += block_start + 1
                if self.boi_max > boi_max:
                    boi_max = self.boi_max
                    boi_max_idx = self.boi_max_idx
                peaks.append(boi_max_idx)
                if self.debug is True:
                    self.peaks.append(boi_max_idx)
                    self.blocks.append((self.boi_start, end))
            changes = changes[1:]
            self.boi = False

        if len(changes) % 2 != 0:
            self.boi = True
            self.boi_start = changes[-1] + block_start
            s = changes[-1] + self.skip_before - 2

            e = len(ma_beat) + self.skip_after - 1  # The last seen element
            boi_max = np.argmax(self.data[s:e])
            self.boi_max_idx = self.boi_start + boi_max
            self.boi_max = self.data[s + boi_max]
            changes = changes[:-1]

        if len(changes) > 1:
            # We've got several changes to test and we
            # are sure changes are even
            assert(len(changes) % 2 == 0)
            # look for midle bois
            boi = changes.reshape(-1, 2)
            for bi in range(boi.shape[0]):
                s = boi[bi, 0] + self.skip_before
                e = boi[bi, 1] + self.skip_before
                if (e - s) > self.qrs_length:
                    peak = np.argmax(self.data[s:e])
                    s = boi[bi, 0] + block_start
                    e = boi[bi, 1] + block_start
                    peak += s + 1
                    peaks.append(peak.ravel())
                    if self.debug is True:
                        self.peaks.append(peak)
                        self.blocks.append((s, e))

        self._consume()
        return np.array(peaks).ravel()

    def qrs_detect_iir(self, data):
        # Filter -> Square
        a = list(data)
        self.filter.filter(a)
        fdata = np.square(a)
        ma_qrs = moving_average(fdata[self.skip_before:-self.skip_after],
                                self.qrs_length)
        ma_beat = moving_average(fdata, self.beat_length)
        z = fdata.mean()
        alpha = self.beta * z
        thr1 = ma_beat + alpha
        boi = np.append(ma_qrs > thr1, False)
        boi = np.insert(boi, 0, False)
        blocks = np.where(boi[1:] - boi[:-1])[0].reshape(-1, 2)
        thr2 = self.qrs_length
        peaks = []
        for bi in range(blocks.shape[0]):
            s = blocks[bi, 0] + self.skip_before
            e = blocks[bi, 1] + self.skip_before
            if (e - s) > thr2:
                peaks.append(s + np.argmax(np.abs(data[s:e])))
        blocks += self.skip_before
        return peaks, blocks


class QRSPredictor(object):
    def __init__(self, sfreq, weights=None, detect_params=None):
        if weights is None:
            weights = np.array([1])

        self.sfreq = sfreq
        self.weights = weights
        self.n_weights = len(weights)

        self.peaks = RingBuffer(4 * len(weights))
        self.missed_peaks = RingBuffer(4 * len(weights))
        self.predictions = RingBuffer(4 * len(weights))

        if detect_params is None:
            detect_params = {}

        detect_params['sfreq'] = sfreq
        self.detector = QRSDetector(**detect_params)

    def _predict_func(self):
        all_peaks = np.union1d(self.peaks.ravel(), self.missed_peaks.ravel())
        n = min(self.n_weights, len(all_peaks) - 1)
        # Get number of samples between elements
        # print n
        d = (all_peaks[-n:] - all_peaks[-n - 1:-1]).ravel()
        # print self.weights[-n:]
        # print d, np.sum(d * self.weights[-n:]), np.sum(self.weights[-n:])
        prediction = (np.sum(d * self.weights[-n:]) /
                      np.sum(self.weights[-n:]))
        prediction = prediction + all_peaks[-1]
        return prediction

    def _check_missed(self, sample, prediction):
        # If prediction is too old... then a peak was missed
        missed_peak = None
        if (prediction is not None and
                sample > prediction + 1.5 * self.detector.beat_length):
            self.missed_peaks.append(prediction, auto=True)
            missed_peak = self.missed_peaks[-1]
            prediction = self._predict_func()
            self.predictions.append(prediction.ravel(), auto=True)
        return missed_peak, prediction

    def predict(self, samples):
        """ Predicts peaks based on previous + new data
            Always returns a prediction.
        """
        peaks, sample = self.detector.add_detect(samples)
        self.sample = sample
        prediction = None
        missed_peak = None
        n_peaks = len(peaks)
        if n_peaks > 0:
            self.peaks.append(peaks, auto=True)
            if len(self.peaks) > 1:
                prediction = self._predict_func()
                if prediction is not None:
                    self.predictions.append([prediction], auto=True)
        else:
            # if no peaks detected, return last prediction
            if len(self.predictions) > 1:
                prediction = self.predictions.ravel()[-1]
            missed_peak, prediction = self._check_missed(sample, prediction)

        # if prediction is not None:
        #     print prediction
        return prediction, peaks, missed_peak, sample


class QRSDephasedPredictor(QRSPredictor):
    def __init__(self, sfreq, n_peaks=200, n_base=20, weights=None,
                 detect_params=None):
        if weights is None:
            weights = np.array([1])

        self.weights = weights
        self.n_weights = len(weights)

        self.sfreq = sfreq
        self.n_peaks = n_peaks

        self.peaks = RingBuffer(4 * n_peaks)
        self.unbias_predictions = RingBuffer(4 * n_peaks)
        self.missed_peaks = RingBuffer(4 * n_peaks)
        self.predictions = RingBuffer(4 * n_peaks)
        # self.distances = RingBuffer(2 * weights)
        if detect_params is None:
            detect_params = {}

        detect_params['sfreq'] = sfreq
        self.detector = QRSDetector(**detect_params)
        self.last = None
        self.n_base = n_base

    def _predict_func(self):
        unbias_prediction = QRSPredictor._predict_func(self)
        if unbias_prediction is not None:
            self.unbias_predictions.append([unbias_prediction], auto=True)
        all_peaks = np.union1d(self.peaks.ravel(), self.missed_peaks.ravel())
        if len(all_peaks) < self.n_base:
            return self.last
        n = min(self.n_peaks, len(all_peaks) - 1)
        d = (all_peaks[-n:] - all_peaks[-n - 1:-1]).ravel()
        if self.last is None:
            mvalue = np.mean(d)
            delay = mvalue * (0.5 * np.random.rand() + 0.25)
            self.last = all_peaks[-1] + delay
            self.initial_delay = delay
            print('Initial delay', delay)

        val = np.random.randint(0, len(d))
        # print len(d), val, d[val], self.sample, self.last, '->', (self.last + d[val])
        self.last = self.last + d[val]
        if self.last < all_peaks[-1] + self.initial_delay:
            # Correct to avoid predicting past peaks
            mvalue = np.mean(d)
            delay = mvalue * (0.5 * np.random.rand() + 0.25)
            print('Correcting from', self.last, 'to', all_peaks[-1] + delay)
            self.last = all_peaks[-1] + delay
        return self.last

    def _check_missed(self, sample, prediction):
        # If prediction is too old... then a peak was missed
        missed_peak = None
        if len(self.unbias_predictions) > 0:
            unbias_prediction = self.unbias_predictions[-1]
            if (sample > unbias_prediction + 1.5 * self.detector.beat_length):
                self.missed_peaks.append(unbias_prediction, auto=True)
                missed_peak = self.missed_peaks[-1]
                prediction = self._predict_func()
                if prediction is not None:
                    self.predictions.append(prediction.ravel(), auto=True)
        return missed_peak, prediction
