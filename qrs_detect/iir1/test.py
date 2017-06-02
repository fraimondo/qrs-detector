import pyiir1 as iir
import matplotlib.pyplot as plt
import numpy as np

srate = 250
lpass = 20
hpass = 0.5

f = 14
w = 2 * np.pi * f

t = np.linspace(0, 1, srate)
y = np.sin(w * t)

slist = list(y)
f = iir.ButterworthBandPass(15, srate, hpass, lpass)
f.reset()
f.filter(slist)
plt.figure()
plt.title("order 15")
plt.plot(slist)

slist = list(y)
f = iir.ButterworthBandPass(8, srate, hpass, lpass)
f.reset()
f.filter(slist)
plt.figure()
plt.title("order 8")
plt.plot(slist)

slist = list(y)
f = iir.ButterworthBandPass(4, srate, hpass, lpass)
f.reset()
f.filter(slist)
plt.figure()
plt.title("order 4")
plt.plot(slist)
