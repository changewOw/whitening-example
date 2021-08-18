import numpy as np # linear algebra

from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

from pycbc.types import TimeSeries as TimeSeries_cbc





data = np.load("0000661522.npy")
print("data shape is ", data.shape)



d0 = TimeSeries(data[0, :], sample_rate=2048)
d1 = TimeSeries(data[1, :], sample_rate=2048)
d2 = TimeSeries(data[2, :], sample_rate=2048)
plot = Plot(d0, d1, d2, separate=True, sharex=True, figsize=[12, 8])
ax = plot.gca()
ax.set_xlim(0, 2)
ax.set_xlabel('Time [s]')
plot.show()




d0_w = d0.whiten(fduration=2)
d1_w = d1.whiten(fduration=2)
d2_w = d2.whiten(fduration=2)

print((d0.whiten(fduration=2, fftlength=2) == d0.whiten(fduration=2)).all())

print(d0.mean())
print(d0.std())

print(d0_w.mean())
print(d0_w.std())



plot = Plot(d0_w, d1_w, d2_w, separate=True, sharex=True, figsize=[12, 8])
ax = plot.gca()
ax.set_xlim(0, 2)
ax.set_xlabel('Time [s]')
plot.show()





