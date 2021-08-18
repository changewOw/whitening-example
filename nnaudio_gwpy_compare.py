from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from copy import deepcopy

from nnAudio.Spectrogram import CQT1992v2


def load_file(fname):
    data = np.load(str(fname))
    d1 = TimeSeries(data[0, :], sample_rate=2048)
    d2 = TimeSeries(data[1, :], sample_rate=2048)
    d3 = TimeSeries(data[2, :], sample_rate=2048)
    return d1, d2, d3

def plot_time_data(d1, d2, d3):
    plot = Plot(d1, d2, d3, separate=True, sharex=True, figsize=[12, 8])
    ax = plot.gca()
    ax.set_xlim(0,2)
    ax.set_xlabel('Time [s]')
    plot.show()

def show_window(window_name, **kwargs):
    plt.subplot(211)
    window = signal.get_window((window_name,0.2), **kwargs)
    plt.plot(window)
    plt.subplot(212)
    window = signal.get_window(window_name, **kwargs)
    plt.plot(window)
    plt.show()


"""
precessing:
1.apply a window function(Tukey-tapered consine window) to suppress spectral leakage
2.white the spectrum
3.bandpass
"""


def precessing(d:TimeSeries, bandpass=False, lf=35, hf=350, copy=True):
    if copy:
        d = deepcopy(d)
    # white_data = d.whiten(window="hanning", fduration=2)
    white_data = d.whiten(window=("tukey", 0.2), fduration=2)
    if bandpass:
        bp_data = white_data.bandpass(lf, hf)
        return bp_data
    else:
        return white_data


def precessing_spectrum(d:TimeSeries, copy=True):
    if copy:
        d = deepcopy(d)
    dq = d.q_transform(fduration=2,whiten=False,
                         logf=True,qrange=(16,32),frange=(30,1024),
                          tres=None, fres=None)
    print("dq", dq.mean(), dq.std(), dq.min(), dq.max(), dq.shape)
    return dq


def plot_spectrum_data(d1:TimeSeries,d2:TimeSeries,d3:TimeSeries):
    d1.plot()
    d2.plot()
    d3.plot()
    plt.show()




if __name__ == '__main__':


    d1, d2, d3 = load_file("000a5b6e5c.npy")
    plot_time_data(d1,d2,d3)

    show_window("tukey", Nx=4096)

    d1_p = precessing(d1)
    d2_p = precessing(d2)
    d3_p = precessing(d3)
    plot_time_data(d1_p,d2_p,d3_p)

    d1_q = precessing_spectrum(d1_p)
    d2_q = precessing_spectrum(d2_p)
    d3_q = precessing_spectrum(d3_p)
    plot_spectrum_data(d1_q, d2_q, d3_q)









