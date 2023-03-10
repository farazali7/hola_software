from scipy.signal import resample_poly


def resample(data, up, down, axis=0):
    """
    Resample a given data array to a sampling rate of up/down.
    :param data: Array of data
    :param up: Int upsampling factor
    :param down: Int downsampling factor
    :param axis: Int axis to resample along
    :return: Data with up/down sampling rate
    """

    resampled = resample_poly(data, up, down, axis=axis)

    return resampled
