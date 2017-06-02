import numpy as np
import pybdf

filename = '../../eeg/data/sound_test_4_arduino.bdf'

bdf = pybdf.bdfRecording(filename)

srate = 2048.0

data = bdf.getData(channels=['Ana1'], trigChan=True)

sound = np.abs(np.squeeze(data['data']))
trig = np.squeeze(data['trigChan'])

sound_thresh = 230000
sound_win = 204

binsound = np.zeros(sound.shape, dtype=np.int)
binsound[sound > sound_thresh] = 1

def goes_up(x, y):
    return x < y

trig_stim_samples = np.where(map(goes_up, trig[:-1], trig[1:]))[0]

def sound_up(x, i):
    return np.mean(x[i:i + sound_win]) > 0.1 * len(sound_win)

i = 0
while i < len(binsound) - 1:
    if goes_up(binsound[i], binsound[i+1]):
        binsound[i] = 1.0
        binsound[i+1:i+sound_win] = 0
        i += sound_win
    else:
        binsound[i] = 0.0
        i += 1

sound_stim_samples = np.where(map(goes_up, binsound[:-1 - sound_win],
                                  binsound[1:-sound_win]))[0]

diff = sound_stim_samples - trig_stim_samples

print 'Max delay (ms)', np.max(diff) / srate * 1000
print 'Min delay (ms)', np.min(diff) / srate * 1000
print 'Mean delay (ms)', np.mean(diff) / srate * 1000
print 'Std delay (ms)', np.std(diff) / srate * 1000
