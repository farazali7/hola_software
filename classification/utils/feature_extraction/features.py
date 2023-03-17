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


def hjorth_mobility(data, var_data, var_axis=1):
    first_deriv = np.gradient(data, 1, axis=0)
    var_grad = var(first_deriv, axis=var_axis)
    mobility = np.sqrt(var_grad / var_data)

    return mobility


def hjorth_complexity(data, var_data, var_axis=1, return_mobility=False):
    mobility = hjorth_mobility(data, var_data=var_data, var_axis=var_axis)

    first_deriv = np.gradient(data, 1, axis=0)
    var_grad = var(first_deriv, axis=var_axis)
    mobility_deriv = hjorth_mobility(first_deriv, var_data=var_grad, var_axis=var_axis)

    complexity = mobility_deriv / mobility

    if return_mobility:
        return complexity, mobility

    return complexity


