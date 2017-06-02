import struct
import numpy as np


def safe_float(x):
    if x.strip() == '':
        return 0.0
    else:
        return float(x)


class GdfReader(object):
    def __init__(self, file_name, **kwargs):
        '''Reads an GDF file at @file_name.
        '''
        self.path = file_name
        self.metadata = dict()
        self.file_handle = open(self.path, 'r')
        self.metadata['version'] = self.file_handle.read(8)
        self.long = lambda: struct.unpack('<q', self.file_handle.read(8))[0]
        self.int = lambda: struct.unpack('<l', self.file_handle.read(4))[0]
        self.short = lambda: struct.unpack('<h', self.file_handle.read(2))[0]
        self.float = lambda: struct.unpack('<f', self.file_handle.read(4))[0]
        self.double = lambda: struct.unpack('<d', self.file_handle.read(8))[0]
        self.str = lambda x: struct.unpack('{}s'.format(x),
                                           self.file_handle.read(x)
                                           )[0].strip().strip('\x00')
        self.metadata['patient'] = self.str(80)
        # self.metadata['reserved'] = self.str(80)
        self.metadata['recording'] = self.str(64)
        self.metadata['location'] = [self.int() for x in range(4)]
        self.metadata['startdate'] = [self.int(), self.int()]
        self.metadata['birthday'] = [self.int(), self.int()]
        self.metadata['header_bytes'] = self.short()
        self.metadata['icd_class'] = self.str(6)
        self.metadata['ep_id'] = self.long()
        self.metadata['reserved_2'] = self.str(6)
        self.metadata['headsize'] = [self.short() for x in range(3)]
        self.metadata['ref_pos'] = [self.float() for x in range(3)]
        self.metadata['gnd_pos'] = [self.float() for x in range(3)]
        self.metadata['ndata'] = self.long()
        self.metadata['ldata'] = [self.int(), self.int()]
        self.metadata['nchannels'] = self.short()
        self.metadata['reserved_3'] = self.str(2)

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
            self.metadata['channels'][i]['prefiltering'] = self.str(64)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['time_offset'] = self.float()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['lowpass'] = self.float()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['highpass'] = self.float()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['notch'] = self.float()
        for i in range(self.nchannels):
            self.metadata['channels'][i]['spr'] = self.int()
            if self.metadata['channels'][i]['spr'] != 1:
                raise ValueError('Cannot read {} samples per record'.format(
                    self.metadata['channels'][i]['spr']
                ))
        for i in range(self.nchannels):
            self.metadata['channels'][i]['dt'] = self.int()
            if self.metadata['channels'][i]['dt'] != 5:
                raise ValueError('Data type {} not supported'.format(
                    self.metadata['channels'][i]['dt']
                ))
        for i in range(self.nchannels):
            self.metadata['channels'][i]['pos'] = \
                [self.float() for x in range(3)]
        for i in range(self.nchannels):
            self.metadata['channels'][i]['ssi'] = self.str(20)
        for i in range(self.nchannels):
            self.metadata['channels'][i]['sampling_rate'] = \
                self.metadata['ldata'][1] / (
                    self.metadata['ldata'][0]
                    * self.metadata['channels'][i]['spr'])

        self.scale = [0 for x in range(self.nchannels)]
        self.dc = [0 for x in range(self.nchannels)]
        for i in range(self.nchannels):
            if (self.metadata['channels'][i]['digital_max'] -
               self.metadata['channels'][i]['digital_min']) == 0:
                print('Warning: channel {} does not define scaling'.format(
                    i))
                self.scale[i] = 1
            else:
                self.scale[i] =  \
                    (self.metadata['channels'][i]['physical_max'] -
                     self.metadata['channels'][i]['physical_min']) / \
                    (self.metadata['channels'][i]['digital_max'] -
                     self.metadata['channels'][i]['digital_min'])
            self.dc[i] = \
                self.metadata['channels'][i]['physical_max'] - \
                (self.scale[i] * self.metadata['channels'][i]
                 ['digital_max'])

        # Prepare buffers for block reads
        for i in range(self.nchannels):
            self.metadata['channels'][i]['rsize'] = \
                self.metadata['channels'][i]['spr']
        self.records_read = 0
        self.ch_to_read = list(range(self.nchannels))
        self.unpack_str = '<' + 'i' * self.nchannels
        self.last_read = self.nchannels

    def read(self, x):
        if self.last_read != x:
            self.unpack_str = '<' + 'i' * x
            self.last_read = x
        return struct.unpack(self.unpack_str, self.file_handle.read(4 * x))

    def _read_sample_block(self, nblocks):
        # print '.',
        # toread = self.metadata['channels'][0]['rsize']
        raw_sample = self.read(self.metadata['nchannels'] * nblocks)
        self.readdata = raw_sample  # * self.scale[channel] + self.dc[channel]
        self.records_read = self.records_read+1
        # print self.records_read
        self.samples_read_in_record = 0

    def read_all_samples(self, nblocks=None, dtype=np.double):
        start = 0
        end = self.metadata['channels'][self.ch_to_read[0]]['rsize']
        if nblocks is None:
            nblocks = self.metadata['ndata']
        elif nblocks > self.metadata['ndata'] + self.records_read:
            raise ValueError('Cannot read past the end of the file')
        samples = np.zeros((len(self.ch_to_read), nblocks),
                           dtype=dtype)
        for block in range(nblocks):
            self._read_sample_block(1)
            for i, v in enumerate(self.ch_to_read):
                samples[i, start:(start + end)] = self.readdata[v]
            start += end
            nblocks -= 1
        for v in self.ch_to_read:
            samples[v, :] = samples[v, :] * self.scale[v] + self.dc[v]
        return samples

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


# if __name__ == "__main__":
#     import time
#     er = EdfReader('tests/test.edf')
#     while er.has_next():
#         print er.read_sample()
#         time.sleep(1.0/er.sampling_rate())
