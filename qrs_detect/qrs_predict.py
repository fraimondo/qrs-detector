import numpy as np

from . import RingBuffer, QRSDetector


class QRSPredictor(object):
    def __init__(self, sfreq, weights=None, detect_params=None):
        if weights is None:
            weights = np.array([1])

        self.sfreq = sfreq
        self.weights = weights
        self.n_weights = len(weights)

        if detect_params is None:
            detect_params = {}

        detect_params['sfreq'] = sfreq
        self.detector = QRSDetector(**detect_params)
        self.reset()

    def reset(self):
        self.peaks = np.array([])
        self.missed_peaks = np.array([])
        self.future = np.array([])
        self.last_peaks_call = 0
        self.last_missed_call = 0
        self.last_beat_length = np.inf
        self.detector.reset()

    @property
    def beta(self):
        return self.detector.beta

    def set_beta(self, value):
        self.detector.set_beta(value)

    @property
    def sample(self):
        return self.detector.nsamples

    def _do_predictions(self, new_peak):
        # Perform a prediction. If new_peak is true, it means that
        # a peak was detected, so update_predictions should be forced to discard
        # based on current sample and not waiting for the detector.

        # Check if we already included a missed peak that might conflict with
        # this new peak

        if new_peak and len(self.missed_peaks) > 0:
            last_missed = self.missed_peaks[-1]
            if (self.peaks[-1] - last_missed) < 0.5 * self.last_beat_length:
                # We have a missed peaks less than half of the beat_rate in
                # the pass. Remove the last missed peak:
                print('Removing bad missed peak', self.missed_peaks[-1])
                self.missed_peaks = self.missed_peaks[:-1]

        n_peaks = min(self.n_weights + 1, len(self.peaks))
        n_missed = min(self.n_weights + 1, len(self.missed_peaks))
        all_peaks = np.union1d(self.peaks[-n_peaks:],
                               self.missed_peaks[-n_missed:])

        n = min(self.n_weights, len(all_peaks) - 1)
        if n == 0:
            return
        # Get number of samples between elements
        d = (all_peaks[-n:] - all_peaks[-n - 1:-1]).ravel()
        # print 'Using distances from', all_peaks[-n:], 'to', all_peaks[-n - 1:-1]
        # print self.weights[-n:]
        # print d, np.sum(d * self.weights[-n:]), np.sum(self.weights[-n:])
        prediction = np.zeros((2), dtype=np.float)
        prediction[0] = (np.sum(d * self.weights[-n:]) /
                         np.sum(self.weights[-n:]))
        prediction[1] = ((np.sum(d[1:] * self.weights[-n:-1]) + prediction[0]) /
                         np.sum(self.weights[-n:]))

        # print self.sample, '=> Using distances from', all_peaks[-n:], 'to', all_peaks[-n - 1:-1], '=', d, '->', prediction,
        prediction[1] += prediction[0]
        self.last_beat_length = prediction[0]
        prediction = prediction + all_peaks[-1]
        # print '=>', prediction
        # Round to avoid differences in floating point errors
        self.future = np.round(prediction)

    def _check_missed(self):
        """ Function to check if a peak was missed by the detector
            should update missed_peaks """
        # self._update_predictions()
        if len(self.future) == 0:
            return
        # last_peak = self.peaks[-1] if len(self.peaks) > 0 else 0
        # last_missed = self.missed_peaks[-1] if len(self.missed_peaks) > 0 else 0
        # Get last peak
        # last = max(last_peak, last_missed)
        # Peek two predictions and check if we dont have time
        # to wait for the peak (we keep 160ms before the peak so we can
        # trigger the sound with 150ms delay)
        if self.sample > self.future[1] - 0.05 * self.sfreq:
            last_prediction = self.future[0]
            # Last prediction becomes a missed peak
            # print self.sample, '=> New Missed', last_prediction, 'last was', last, '(', self.sample - last, ')'
            print('Missed peak', last_prediction)
            self.missed_peaks = np.append(self.missed_peaks, last_prediction)
            self._do_predictions(new_peak=False)

    def add_predict(self, samples):
        """add samples to predict """
        peaks = self.detector.add_detect(samples)
        n_peaks = len(peaks)
        if n_peaks > 0:
            self.peaks = np.append(self.peaks, peaks)
            if len(self.peaks) > 1:
                # print self.sample, '=> New Peaks', peaks
                self._do_predictions(new_peak=True)
        else:
            self._check_missed()

    def get_peaks(self):
        """return detected and missed peaks since last peaks call"""
        peaks = self.peaks[self.last_peaks_call:]
        missed = self.missed_peaks[self.last_missed_call:]
        self.last_peaks_call += len(peaks)
        self.last_missed_call += len(missed)
        return peaks, missed

    def next(self):
        """return next peak"""
        # Get first future prediction (if available)
        next_pred = None
        if len(self.future) > 0:
            next_pred = self.future[0]
        return next_pred


class QRSDephasedPredictor(QRSPredictor):
    def __init__(self, sfreq, n_peaks=200, n_base=20, weights=None,
                 detect_params=None):
        super(QRSDephasedPredictor, self).__init__(
            sfreq=sfreq, weights=weights, detect_params=detect_params)

        self.n_peaks = n_peaks
        self.n_base = n_base

    def reset(self):
        super(QRSDephasedPredictor, self).reset()
        self.initial_delay = None
        self.next_pred = None

    def _do_predictions(self, new_peak):
        super(QRSDephasedPredictor, self)._do_predictions(new_peak)
        n_peaks = min(self.n_peaks + 1, len(self.peaks))
        n_missed = min(self.n_peaks + 1, len(self.missed_peaks))
        all_peaks = np.union1d(self.peaks[-n_peaks:],
                               self.missed_peaks[-n_missed:])
        n = min(self.n_peaks - 1, len(all_peaks) - 1)

        # Wait untill we have at least n_base peaks
        if n <= self.n_base:
            return
        # Get number of samples between elements
        d = (all_peaks[-n:] - all_peaks[-n - 1:-1]).ravel()
        if self.initial_delay is None:
            mvalue = np.mean(d)
            self.initial_delay = mvalue * (0.5 * np.random.rand() + 0.45)

        # Do we have a last dephased peak?
        if self.next_pred is None:
            # If not, create a dephased peak with the last delay
            last_diff = self.initial_delay
        else:
            # Get the error based on last prediction and last peak
            last_diff = self.next_pred - all_peaks[-1]

        # Last peak was to close to the real peak?
        if abs(last_diff) < 0.2 * self.sfreq:
            # Add some random delay
            delta = np.mean(d) * (0.5 * np.random.rand() + 0.25)
            print('Preventing phase beat, delaying by', delta)
            last_diff += delta

        # Take a random interval between peaks
        val = np.random.randint(0, len(d))

        self.next_pred = all_peaks[-1] + last_diff + d[val]
        if self.next_pred < all_peaks[-1] + self.initial_delay:
            # Correct to avoid predicting past peaks
            mvalue = np.mean(d)
            delay = mvalue * (0.5 * np.random.rand() + 0.25)
            print('Correcting from', self.next_pred, 'to', all_peaks[-1] + delay)
            self.next_pred = all_peaks[-1] + delay
        self.next_pred = np.round(self.next_pred)

    def next(self):
        """return next peak"""
        # Get first future prediction (if available)
        return self.next_pred
