import numpy as np

import u3
import traceback
from datetime import datetime

from argparse import ArgumentParser
from scipy import io as sio

import sys
# sys.path.append('../../../../fieldtrip/realtime/src/buffer/python')
# sys.path.append('../../fieldtrip/realtime/src/buffer/python')
sys.path.append('/Users/fraimondo/dev/fieldtrip/realtime/src/buffer/python')
import FieldTrip as ft

parser = ArgumentParser(description='Stream labjac to fieltrip buffer')
parser.add_argument('--host', metavar='host', type=str, nargs=1,
                    default='localhost',
                    help='fieldtrip buffer host (default localhost)')
parser.add_argument('--port', metavar='port', type=int, nargs=1,
                    default=1972,
                    help='fieldtrip buffer port (default 1972)')

args = parser.parse_args()
host = args.host
port = args.port

print('Streaming labjack to {}:{}'.format(host, port))

d = u3.U3()
#
# to learn the if the U3 is an HV
d.configU3()
#
# For applying the proper calibration to readings.
d.getCalibrationData()

# Set the FIO0 and FIO1 to Analog, FIO4 to Digital
d.configIO(FIOAnalog=0x33)
#
# print('configuring U3 stream'
# d.streamConfig(
#     NumChannels=2,
#     PChannels=[0, 4],
#     NChannels=[31, 5],
#     Resolution=3,
#     ScanFrequency=8192/2)
#
d.streamConfig(
    NumChannels=2,
    PChannels=[0, 193],
    NChannels=[31, 31],
    Resolution=3,
    ScanFrequency=250,
    SamplesPerPacket=4)

ftc = ft.Client()

ftc.connect(host, port)
ftc.putHeader(
    2,
    250,
    ft.DATATYPE_FLOAT32,
    ['AIN0', 'DIN6']

)

missed = 0
dataCount = 0
packetCount = 0
data_samps = np.array([])
try:
    print('start stream', end=' ')
    d.streamStart()
    start = datetime.now()
    print(start)

    for r in d.streamData():
        if r is not None:
            # if r['errors'] != 0:
            #     print('Error: {} ; ' % r['errors'], datetime.now()
            # if r['numPackets'] != d.packetsPerRequest:
            #     print('----- UNDERFLOW : {} : ' % \
            #         r['numPackets'], datetime.now()
            # if r['missed'] != 0:
            #     missed += r['missed']
            #     print('+++ Missed ', r['missed']
            print('packet length {} {} {}'.format(
                len(r['AIN0']), len(r['AIN193']), r['AIN193']))
            samp = np.c_[r['AIN0'], map(lambda x: x[0] >> 6 & 0x1, r['AIN193'])]
            ftc.putData(samp.astype(np.float32))
            data_samps = np.append(data_samps, samp)

            # Comment out these prints and do something with r
            # print('Average of' , len(r['AIN0']), 'AIN0,', len(r['AIN4']), 'AIN4 reading(s):',
            # print(sum(r['AIN0']) / len(r['AIN0']) , "," , sum(r['AIN4'])/len(r['AIN4'])

            dataCount += 1
            packetCount += r['numPackets']
except:
    print(''.join(i for i in traceback.format_exc()))
finally:
    stop = datetime.now()
    d.streamStop()
    print('stream stopped.')
    d.close()

    sampleTotal = packetCount * d.streamSamplesPerPacket

    scanTotal = sampleTotal / 3  # sampleTotal / NumChannels
    print('{} requests with {} packets per request '
          'with {} samples per packet = {} samples total.'.format(
            dataCount, (float(packetCount) / dataCount),
            d.streamSamplesPerPacket, sampleTotal))
    print('{} samples were lost due to errors.'.format(missed))
    sampleTotal -= missed
    print('Adjusted number of samples = {}'.format(sampleTotal))

    runTime = (stop-start).seconds + float((stop-start).microseconds)/1000000
    print('The experiment took {} seconds.'.format(runTime))
    print('Scan Rate : {} scans / {} seconds = {} Hz'.format(
        scanTotal, runTime, float(scanTotal)/runTime))
    print('Sample Rate : {} samples / {} seconds = {} Hz'.format(
        sampleTotal, runTime, float(sampleTotal) / runTime))

    sio.savemat('data_samps.mat', {'data': data_samps})
