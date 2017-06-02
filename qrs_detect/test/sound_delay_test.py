import time
import parallel

import numpy as np
from psychopy import prefs
#prefs.general['audioLib'] = ['pygame']
prefs.general['audioLib'] = ['pyo']
prefs.general['audioDriver'] = ['jack']

from psychopy import sound

beep = sound.Sound(
    value='C', secs=0.1, octave=4,
    sampleRate=44100, bits=16, name='standard',
    autoLog=True)

parallel_port = parallel.Parallel()

running = True
parallel_port.setData(0x00)
while running:
    try:
        beep.play()
        parallel_port.setData(0xFF)
        time.sleep(0.1)
        parallel_port.setData(0x00)
        time.sleep(np.random.rand() + 0.2)
    except KeyboardInterrupt:
        print 'Quitting'
        running = False
