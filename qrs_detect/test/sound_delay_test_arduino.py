import time
import parallel

import numpy as np

parallel_port = parallel.Parallel()

running = True
parallel_port.setData(0x00)
i = 0
while running:
    try:
        parallel_port.setData(0x02 + (0x04 if i % 4 == 0 else 0))
        time.sleep(1/250.0)
        parallel_port.setData(0x00)
        time.sleep(60.0/200 + np.random.rand()/2.0)
        i += 1
    except KeyboardInterrupt:
        print 'Quitting'
        running = False
