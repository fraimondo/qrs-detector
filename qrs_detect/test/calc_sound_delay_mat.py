import numpy as np

from scipy import io as sio

period = '128'

filename = '/Volumes/Big/heart/mri/labjack_test/data_samps_{}.mat'.format(period)

start = 0
end = 0

mc = sio.loadmat(filename)

data = np.reshape(mc['data'], [-1, 2])

sfreq = 250.0


sound = np.abs(np.squeeze(data[start:, 0]))
trig = np.squeeze(data[start:, 1])

sound_thresh = 1.0
sound_win = sfreq / 4

binsound = np.zeros(sound.shape, dtype=np.int)
binsound[sound > sound_thresh] = 1

def goes_up(x, y):
    return x < y

trig_stim_samples = np.where(map(goes_up, trig[:-1], trig[1:]))[0]

def sound_up(x, i):
    return np.mean(x[i:i + sound_win]) > 0.7

i = 0
while i < len(binsound) - 1:
    if goes_up(binsound[i], binsound[i+1]):
        binsound[i] = 1.0
        binsound[i+1:i+(sfreq/10)] = 0
        i += (sfreq/10)
    else:
        binsound[i] = 0.0
        i += 1

sound_stim_samples = np.where(map(goes_up, binsound[:-1 - sound_win],
                                  binsound[1:-sound_win]))[0]

delays = sound_stim_samples - trig_stim_samples

print 'Max delay (s)', np.max(delays) / sfreq
print 'Min delay (s)', np.min(delays) / sfreq
print 'Mean delay (s)', np.mean(delays) / sfreq
print 'Std delay (s)', np.std(delays) / sfreq

import matplotlib.pyplot as plt


fig = plt.figure()
plt.hist(delays / sfreq * 1000, bins=np.max(delays) - np.min(delays))
fig.suptitle('{}: Sound - Trigger delays'.format(period))
plt.xlabel('Delay in ms')
plt.ylabel('# of elements')

fig.savefig('delays_{}.png'.format(period))
