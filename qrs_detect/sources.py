from __future__ import print_function
import sys
sys.path.append('/Users/fraimondo/dev/fieldtrip/realtime/src/buffer/python')
sys.path.append('/home/fraimondo/dev/fieldtrip/realtime/src/buffer/python')
import FieldTrip as ft

import numpy as np

import psychopy.logging as logger
logger.console.setLevel(logger.INFO)

from datetime import datetime


class DataSource(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def read_samples(self):
        raise NotImplementedError('Must subclass')

    def get_data(self):
        raise NotImplementedError('Must subclass')


class FtBufferQRS(DataSource):
    def __init__(self, ecg_chan=0, snd_chan=1, dig_chan=2, const_chan=None,
                 host='localhost', port=1972):
        self.ecg_chan = ecg_chan
        self.const_chan = const_chan
        self.snd_chan = snd_chan
        self.dig_chan = dig_chan
        if const_chan is not None:
            self._index = [self.ecg_chan, self.const_chan, self.snd_chan,
                           self.dig_chan]
        else:
            self._index = [self.ecg_chan, self.snd_chan, self.dig_chan]
        self.host = host
        self.port = port

    def __enter__(self):
        self.ftc = ft.Client()
        self.ftc.connect(self.host, self.port)
        self.header = None
        self._read_samples = 0
        return self

    def __exit__(self, type, value, traceback):
        self.ftc.disconnect()

    def sample_freq(self):
        return self.header.fSample if self.header is not None else 1.0

    def read_samples(self):
        return self._read_samples

    def get_data(self):
        while True:
            header = self.ftc.getHeader()
            if header is None:
                logger.warning('Header could not be read')
                yield None
            else:
                if self.header is None:
                    self._read_samples = header.nSamples
                    self.header = header
                    yield None
                else:
                    self.header = header
                    if self.header.nSamples > self._read_samples:
                        data = self.ftc.getData(
                            [self._read_samples, self.header.nSamples - 1])
                        self._read_samples = self.header.nSamples
                        yield data[:, self._index]
                    else:
                        yield None

try:
    import u3
except:
    print('Cannot import UR, LabJackQRS source will not work')


class LabjackQRS(DataSource):
    def __init__(self):
        self._read_samples = 0
        self._new_samples = 0

    def __enter__(self):
        self._labjack_config()
        return self

    def __exit__(self, type, value, traceback):
        self.lj.streamStop()
        print('stream stopped.')
        self.lj.close()

    def _labjack_config(self):
        self.lj = u3.U3()
        # to learn the if the U3 is an HV
        self.lj.configU3()
        # For applying the proper calibration to readings.
        self.lj.getCalibrationData()

        # Set the FIO0 and FIO1 to Analog, FIO4 to Digital
        self.lj.configIO(FIOAnalog=0x3F)
        #
        logger.info('configuring U3 stream')
        self.lj.streamConfig(
            NumChannels=3,
            PChannels=[4, 0, 193],
            NChannels=[5, 1, 31],
            Resolution=3,
            ScanFrequency=250,
            SamplesPerPacket=12)

        self.lj.streamStart()

    def sample_freq(self):
        return 250.0  # ScanFrequency

    def read_samples(self):
        return self._read_samples

    def get_data(self):
        for r in self.lj.streamData():
            if r is None:
                yield None
            else:
                if r['errors'] != 0:
                    logger.error(
                        'Error: {} : {}'.format(r['errors'], datetime.now()))
                if r['numPackets'] != self.lj.packetsPerRequest:
                    logger.error(
                        '----- UNDERFLOW : {} : {}'.format(
                            r['numPackets'], datetime.now()))
                if r['missed'] != 0:
                    logger.error("+++ Missed {}".format(r['missed']))
                new_data = np.c_[r['AIN4'], r['AIN0'],
                                 map(lambda x: x[0] >> 6 & 0x1, r['AIN193'])]
                # logger.info('Received {}'.format(len(new_data)))
                self._new_samples = len(new_data)
                self._read_samples += self._new_samples
                yield new_data
