import numpy as np
import pywt
from numbers import Number


def rms(data):
    for x in data:
        if not isinstance(x, (int, float)):
            return 'NA'
    return np.sqrt(np.mean(np.array(data)**2, 2))


def mav(data):
    return np.mean(np.abs(data), 2)


def var(data):
    return np.var(data, 2)


def dwt(data, familly='db7', level=3, axis=2):
    return pywt.wavedec(data, familly, level=level, axis=axis)

