import numpy as np
import pywt


def rms(data, axis=1):
    return np.sqrt(np.mean(np.array(data)**2, axis))


def mav(data, axis=1):
    return np.mean(np.abs(data), axis)


def var(data, axis=1):
    return np.var(data, axis)


def dwt(data, family='db7', level=3, axis=2):
    return pywt.wavedec(data, family, level=level, axis=axis)

