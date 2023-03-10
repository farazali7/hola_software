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


def hjorth_mobility(data):
    first_deriv = np.gradient(data, 1, axis=0)
    var_grad = var(first_deriv)
    mobility = np.sqrt(var_grad / first_deriv)

    return mobility


def hjorth_complexity(data, precomputed_mobility=None):
    mobility = precomputed_mobility if not None else hjorth_mobility(data)
    mobility_deriv = hjorth_mobility(np.gradient(data, 1, axis=0))
    complexity = mobility_deriv / mobility

    return complexity


