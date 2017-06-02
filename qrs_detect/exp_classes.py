# -*- coding: utf-8 -*-
from __future__ import print_function

import time

import numpy as np

from psychopy import visual, logging


def text(win, string=None, image=None, alignHoriz='center'):
    if image is not None:
        image.draw()
    else:
        if isinstance(string, str) or isinstance(string, unicode):
            text = visual.TextStim(win, text=string, color='black', height=20,
                                   alignHoriz=alignHoriz)
        else:
            text = string
        text.draw()
    win.flip()


class ParallelController():
    def __init__(self):
        self.use_parallel = False
        try:
            import parallel
            self.use_parallel = True
            self.parallel_port = parallel.Parallel()
        except:
            print('Not using parallel port, stimulation will be simulated')
        self.pulse_sent = False

    def send_pulse(self, value):
        if self.use_parallel:
            self.parallel_port.setData(value)
            logging.exp('Stim 0x{:X}'.format(value))
            self.pulse_sent = True
        else:
            print('SIM: parallel pulse 0x{:X}'.format(value))

    def end_pulse(self):
        if self.use_parallel:
            self.parallel_port.setData(0x00)
            self.pulse_sent = False


class TactileStimController():

    def __init__(self, delay=0):
        self.delay = 0
        self.use_tactile = False
        self.stim_sent = False
        self.last_stim = 0
        try:
            import serial
            self.port = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
            self.stim_string = 'xC5C6C7C8C1C4B5B6B7B8ey\n'
            self.clear()
            self.use_tactile = True
            self.delay = delay
        except:
            print('Not using tactile stimulation')

    def clear(self):
        if self.use_tactile:
            self.port.write('xy\n')

    def set_stimulate_string(self, value):
        self.stim_string = value

    def stimulate(self):
        if self.use_tactile:
            self.port.write(self.stim_string)
            self.stim_sent = True
            self.last_tact_stim = time.time()

    def get_delay(self):
        return self.delay

    def end(self):
        if self.use_tactile:
            now = time.time()
            if (now - self.last_stim) > 0.3:
                self.stim_sent = False
                self.clear()


class Block(object):
    """ Definition of a block
    Parameters:
        beats : number of beats to complete block
        sync_predictor : If True, is sync predcitor, else use osync
        block_kind : string identifying the block kind
        block_kind_code : Value to add to parallel pulse
            Bits 0-2 are for stimulation
            Bit 6-7 are for start - end codes
            Only bits 3-5 available. Possible values are:
                - 0x00
                - 0x08
                - 0x10
                - 0x18
                - 0x20
                - 0x28
                - 0x30
                - 0x38
        stimulate : which kind of stimulation to use
            False : No stimulation
            True : Standard except the last one (MRI)
            Float : Deviant factor to use (EEG)
        wait_for_key : weither to wait for key press or not (True or False)
        timeout : Number of seconds to wait for key press (False is disabled)
        key_list : Possible keys to wait
        subject_text : Text to show to the subject.
            If stimulate is True, then this text is showed only at the end
        exp_win : Experimenter window



    """
    def __init__(self, beats, sync_predictor, block_kind, block_kind_code,
                 stimulate, wait_for_key, timeout, parallel, exp_win,
                 tactile=None, key_list=None, subject_text=None, subj_win=None):
        self.beats = beats
        self.sync_predictor = sync_predictor
        self.block_kind = block_kind
        self.block_kind_code = block_kind_code
        self.stimulate = stimulate

        self.wait_for_key = wait_for_key
        self.timeout = timeout
        self.key_list = key_list
        self.subject_text = subject_text

        self.exp_win = exp_win
        self.subj_win = subj_win
        self.parallel = parallel
        self.tactile = tactile

        if self.tactile is not None:
            self.delay = self.tactile.get_delay()
        else:
            self.delay = 0

        logging.exp('{} ({}) Created with delay {}'.format(
            self.block_kind, self.block_kind_code, self.delay))

        # Internal variables
        self.beats_done_ = 0
        self.waiting_key_ = False
        self.started_ = 0

        if isinstance(self.stimulate, float):
            self.stims_ = 0x3 * np.ones((self.beats), dtype=np.int)
            i = 1
            while i < self.beats:
                dvt = np.random.rand() < self.stimulate
                if dvt:
                    self.stims_[i] = 0x7
                    i = i + 2
                else:
                    i = i + 1

    def _exp_text(self):
        status = ''
        if self.waiting_key_:
            status = 'Waiting for key ({})'.format(' '.join(self.key_list))
        to_show = ('{}\nBlock Type: {}\nBeats {} of {} ({} left)\n\n{}\n{}\n'
                   'Current Beta: {}'.format(
                       self.prepend_exp_text_, self.block_kind,
                       self.beats_done_, self.beats,
                       self.beats - self.beats_done_, self.append_exp_text,
                       status, self.predictor_.beta))
        return to_show

    def reset(self):
        # This is the list of upcoming predicted peaks
        self.predictions_ = np.array([])  # Create empty prediction list

        # This is the sample where the next peak is comming
        self.next_peak_ = np.inf  # Set next peak to never

        # This is the sample where the last peak was stimulated
        self.last_peak_ = 0  # Set last peak to first sample

        self.parallel.send_pulse(0xC0 + self.block_kind_code)  # Reset pulse
        if self.tactile is not None:
            self.tactile.clear()
        logging.exp('{} ({}) Reseted'.format(
            self.block_kind, self.block_kind_code))

    def start(self, sync_predictor, osync_predictor, idx_block, n_blocks,
              last_condition):
        # Save predictor being used
        if self.sync_predictor:
            self.predictor_ = sync_predictor
            self.other_predictor_ = osync_predictor
        else:
            self.predictor_ = osync_predictor
            self.other_predictor_ = sync_predictor

        self.reset()
        self.started_ = time.time()
        self.parallel.send_pulse(0x80 + self.block_kind_code)  # Start pulse
        logging.exp('{} ({}) Started'.format(
            self.block_kind, self.block_kind_code))

        self.prepend_exp_text_ = 'Block {} of {}\n\n'.format(
            idx_block, n_blocks)
        self.append_exp_text = 'Last block condition: {}'.format(last_condition)
        text(self.exp_win, string=self._exp_text(), alignHoriz='right')
        if self.stimulate is not True:
            if isinstance(self.subject_text, visual.ImageStim):
                text(self.subj_win, image=self.subject_text)
            else:
                text(self.subj_win, string=self.subject_text)
            text(self.subj_win, string=self.subject_text)

    def add_sample(self, data):
        predicted_peak = 0
        detected_peak = 0
        missed_peak = 0
        # Do it with the other predictor
        self.other_predictor_.add_predict(np.copy(data).tolist())
        self.other_predictor_.next()
        self.other_predictor_.get_peaks()

        # Use values from the one we want
        self.predictor_.add_predict(np.copy(data).tolist())
        predicted = self.predictor_.next()

        if predicted is not None:
            if predicted != self.last_peak_:
                if predicted not in self.predictions_:
                    predicted_peak = predicted
                    self.predictions_ = np.append(self.predictions_,
                                                  predicted_peak)
                    # print(self.predictions_)
                    if self.next_peak_ == np.inf:
                        # print('Updated next peak {}'.format(predicted_peak))
                        self.next_peak_ = predicted_peak

        # Check if next peak will happen in next data read
        # XXX: If delay is detected, it should be earlier
        if (self.last_peak_ != self.next_peak_ and
                # self.next_peak_ <= self.predictor_.sample):
                self.next_peak_ <= (self.predictor_.sample +
                                    self.predictor_.sfreq * self.delay)):
            # Update variables and stimulate

            # Last peak is this one
            self.last_peak_ = self.next_peak_

            # Remove this one from the predictions
            self.predictions_ = self.predictions_[1:]

            # Get next predicted peak
            if len(self.predictions_) > 0:
                self.next_peak_ = self.predictions_[0]
            else:
                self.next_peak_ = np.inf
            # print('Stim -> Next {}'.format(self.next_peak_))
            self._stimulate()

        # Now update values for UI
        detected, missed = self.predictor_.get_peaks()
        if len(detected) > 0:
            detected_peak = detected[0]
        if missed is not None and len(missed > 0):
            missed_peak = missed
        start_sample = self.predictor_.sample
        # print('Sample {}'.format(start_sample))
        return predicted_peak, detected_peak, missed_peak, start_sample

    def _stimulate(self):
        if self.beats_done_ < self.beats:
            self.beats_done_ += 1
            if self.stimulate is True:
                if self.beats_done_ == self.beats:
                    self.parallel.send_pulse(0x7 + self.block_kind_code)
                else:
                    self.parallel.send_pulse(0x3 + self.block_kind_code)
                if self.tactile is not None:
                    self.tactile.stimulate()
            elif self.stimulate is False:
                self.parallel.send_pulse(0x1 + self.block_kind_code)
            else:
                self.parallel.send_pulse(
                    self.stims_[self.beats_done_ - 1] + self.block_kind_code)
                if self.tactile is not None:
                    self.tactile.stimulate()

        text(self.exp_win, string=self._exp_text(), alignHoriz='right')

    def finished(self, keys_pressed):
        """ Given the list of keys pressed, returns if the block is finished
        and which condition should be stored as the block result """
        result = False  # If true, then block is finished
        condition = None  # Condition to save as block results
        if self.beats_done_ >= self.beats:
            if self.wait_for_key is False:
                result = True
                condition = 'Done'  # Done but no response was necesary
            else:
                # import pdb; pdb.set_trace()
                if self.waiting_key_ is False:
                    self.waiting_key_ = True
                    self.waiting_start_ = time.time()
                    if self.stimulate is True:
                        if isinstance(self.subject_text, visual.ImageStim):
                            text(self.subj_win, image=self.subject_text)
                        else:
                            text(self.subj_win, string=self.subject_text)
                    text(self.exp_win, string=self._exp_text(),
                         alignHoriz='right')
                elif any([x for x in keys_pressed if x in self.key_list]):
                    self.waiting_key_ = False
                    result = True
                    # Condition here is all the key pressed (normally should be
                    # only one key, but just in case...)
                    condition = ''.join(
                        [x for x in keys_pressed if x in self.key_list])
                elif self.timeout is not False:
                    if time.time() - self.waiting_start_ > self.timeout:
                        result = True
                        condition = 'Timeout'

        if result is True:
            text(self.subj_win, string='', image=None)
            self.parallel.send_pulse(0x40 + self.block_kind_code)  # End pulse
            logging.exp('{} ({}) Finished -> {}'.format(
                self.block_kind, self.block_kind_code, condition))
            logging.flush()
        return result, condition

    def elapsed(self):
        return time.time() - self.started_

    def __repr__(self):
        desc = '<{}: beats out of {} ({} left)>'.format(
            self.block_kind, self.beats_done_, self.beats,
            self.beats - self.beats_done_)
        return desc
