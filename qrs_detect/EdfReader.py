from __future__ import print_function
import struct
import numpy as np


def safe_float(x):
    sx = x.decode('ascii').strip()
    if sx.strip() == '':
        return 0.0
    else:
        return float(sx)


class EdfReader(object):
    def __init__(self, file_name, **kwargs):
        '''Reads an EDF file at @file_name.
        '''
        self.path = file_name
        self.metadata = dict()
        self.file_handle = open(self.path, 'rb')
        self.metadata['version'] = self.file_handle.read(8)
        self.long = lambda: int(self.file_handle.read(8))
        self.int = lambda: int(self.file_handle.read(4))
        self.double = lambda: safe_float(self.file_handle.read(8))
        self.str = lambda x: self.file_handle.read(x).strip()
        self.metadata['patient'] = self.str(80)
        self.metadata['recording'] = self.str(80)
        self.metadata['startdate'] = self.str(8)
        self.metadata['starttime'] = self.str(8)
        self.metadata['header_bytes'] = self.long()
        self.metadata['reserved'] = self.str(44)
        self.metadata['ndata'] = self.long()

        self.metadata['ldata'] = self.long()
        self.metadata['nchannels'] = self.int()

        self.nchannels = self.metadata['nchannels']

        self.metadata['channels'] = [dict() for x in range(self.nchannels)]

        for i in range(self.nchannels):
            self.metadata['channels'][i]['label'] = self.str(16)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['type'] = self.str(80)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['dimension'] = self.str(8)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['physical_min'] = self.double()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['physical_max'] = self.double()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['digital_min'] = self.double()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['digital_max'] = self.double()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['prefiltering'] = self.str(80)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['sampling_rate'] = self.long()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['reserved'] = self.str(32)

        self.scale = [0 for x in range(self.nchannels)]
        self.dc = [0 for x in range(self.nchannels)]
        for i in range(self.nchannels):
            if (self.metadata['channels'][i]['digital_max'] -
               self.metadata['channels'][i]['digital_min']) == 0:
                print('Warning: channel {} does not define scaling'.format(i))
                self.scale[i] = 1
            else:
                self.scale[i] =  \
                    (self.metadata['channels'][i]['physical_max'] -
                     self.metadata['channels'][i]['physical_min']) / \
                    (self.metadata['channels'][i]['digital_max'] -
                     self.metadata['channels'][i]['digital_min'])
            self.dc[i] = \
                self.metadata['channels'][i]['physical_max'] - \
                (self.scale[i] * self.metadata['channels'][i]['digital_max'])
        self.readdata = [[] for x in range(self.nchannels)]
        # Prepare buffers for block reads
        for i in range(self.nchannels):
            self.readdata[i] = \
                [0 for x in range(self.metadata['channels'][i]['sampling_rate']
                 * self.metadata['ldata'])]

        self.records_read = 0
        self.samples_read_in_record = self.sampling_rate()
        self.ch_to_read = list(range(self.nchannels))

    def _read_sample_block(self):
        print('Reading block',)
        for channel in range(self.nchannels):
            toread = self.metadata['channels'][channel]['sampling_rate'] \
                * self.metadata['ldata']
            for sample in range(toread):
                raw_sample = struct.unpack('<h', self.file_handle.read(2))[0]
                self.readdata[channel][sample] = \
                    raw_sample  # * self.scale[channel] + self.dc[channel]
        self.records_read = self.records_read+1
        print(self.records_read)
        self.samples_read_in_record = 0

    def ensure_sample_block(self):
        if self.samples_read_in_record == self.sampling_rate():
            self._read_sample_block()

    def read_all_samples(self, nblocks=None):
        start = 0
        end = self.sampling_rate(self.ch_to_read[0])
        if nblocks is None:
            nblocks = self.metadata['ndata']
        samples = np.zeros((len(self.ch_to_read), nblocks *
                           self.sampling_rate(self.ch_to_read[0])),
                           dtype=np.double)
        while self.has_next() and nblocks > 0:
            print('.',)
            self._read_sample_block()
            for i, v in enumerate(self.ch_to_read):
                samples[i, start:(start + end)] = np.double(
                    self.readdata[v][:]) * self.scale[v] + self.dc[v]
            start += end
            nblocks -= 1
        return samples

    ''' Reads a sample from each channel. If needed, reads a sample block
    from the file.
    Asumes every channel has the same sampling rate
    '''
    def read_sample(self):
        if self.samples_read_in_record == self.sampling_rate():
            self._read_sample_block()
        sample = [0.0 for x in self.ch_to_read]
        for i, v in enumerate(self.ch_to_read):
            sample[i] = self.readdata[v][self.samples_read_in_record]
        self.samples_read_in_record = self.samples_read_in_record+1
        return sample

    def has_next(self):
        return self.records_read != self.metadata['ndata']

    def set_channels(self, chans):
        self.ch_to_read = chans

    def get_read_channels(self):
        return len(self.ch_to_read)

    def get_nchannels(self):
        return self.nchannels

    def labels(self):
        return [self.metadata['channels'][i]['label']
                for i in self.ch_to_read]

    def sampling_rate(self, nchannel=0):
        return self.metadata['channels'][nchannel]['sampling_rate']

    def close(self):
        self.file_handle.close()


if __name__ == "__main__":
    import time
    er = EdfReader('tests/test.edf')
    while er.has_next():
        print(er.read_sample())
        time.sleep(1.0/er.sampling_rate())
