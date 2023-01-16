import numpy as np
import pywt


def rms(data):
    return np.sqrt(np.mean(np.array(data)**2, 2))


def mav(data):
    return np.mean(np.abs(data), 2)


def var(data):
    return np.var(data, 2)


def dwt(data, family='db7', level=3, axis=2):
    return pywt.wavedec(data, family, level=level, axis=axis)

