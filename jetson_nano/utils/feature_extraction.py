from classification.src.utils.preprocessing import window_data
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from scipy.signal import spectrogram as sp_spectogram
from sklearn.decomposition import PCA as sk_PCA
import numpy as np
import pywt

def spectogram(data, sampling_freq, window, nperseg, n_overlap, axis=-1):
    f, t, Sxx = sp_spectogram(x=data, fs=sampling_freq, window=window, nperseg=nperseg, noverlap=n_overlap, axis=axis,
                              mode='magnitude')

    return f, t, Sxx


def PCA(data, n_components):
    pca = sk_PCA(n_components=n_components)
    comps = pca.fit_transform(data)

    return comps

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



def get_features_2(data):
    """
    Compute RMS, MAV, VAR, mDWT, Mobility, and Complexity.
    :param emg_data: Array of EMG data
    :param grasp_labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    window_size = 60
    window_overlap_size = 30
    combine_channels = False
    standardize = False

    emg_windows = window_data(data, window_size, window_overlap_size)

    # Compute features
    axis = combine_channels + 1

    # RMS
    rms_data = rms(emg_windows, axis=axis)

    # Mean Absolute Value (MAV)
    mav_data = mav(emg_windows, axis=axis)

    # Variance
    var_data = var(emg_windows, axis=axis)

    # Hjorth Mobility & Complexity
    hjorth_mobility_data, hjorth_complexity_data = hjorth_complexity(emg_windows,
                                                                     var_data=var_data,
                                                                     var_axis=axis,
                                                                     return_mobility=True)

    # marginal Discrete Wavelet Transform (mDWT)
    mdwt_data = dwt(emg_windows, family='db7', level=2, axis=axis)
    mdwt_data_means = [np.nanmean(mdwt, axis) for mdwt in mdwt_data]

    raw_features = [rms_data, mav_data, var_data, hjorth_mobility_data, hjorth_complexity_data] + mdwt_data_means

    # Standardize features
    ss = StandardScaler() if standardize else MaxAbsScaler()
    scaled_features = [ss.fit_transform(feature) for feature in raw_features]

    emg_features = np.concatenate(scaled_features, axis=1)

    return emg_features


def get_features(data):
    """
    Compute top PCs from spectrogram with Hamming window applied and reshape with most important PCs in middle.
    :param data: Array of EMG data
    :param labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    window_size = 60
    window_overlap_size = 30
    sampling_freq = 250

    emg_windows = window_data(data, window_size, window_overlap_size)

    # Apply hamming window and get spectogram
    window = 'hamming'
    f, t, Sxx = spectogram(data=emg_windows, sampling_freq=sampling_freq, window=window, nperseg=20,
                           n_overlap=12, axis=1)

    Sxx = np.transpose(Sxx, (0, 3, 1, 2))

    # Normalize to 0 to 1 range & PCA
    n_components = 25
    features = np.zeros(shape=(Sxx.shape[0], 5, 5, Sxx.shape[-1]))
    # Most important in center (r,c)
    spiral_idxs = [(0, (2, 2)), (1, (1, 2)), (2, (1, 3)), (3, (2, 3)), (4, (3, 3)),
                   (5, (3, 2)), (6, (3, 1)), (7, (2, 1)), (8, (1, 1)), (9, (0, 1)),
                   (10, (0, 2)), (11, (0, 3)), (12, (0, 4)), (13, (1, 4)), (14, (2, 4)),
                   (15, (3, 4)), (16, (4, 4)), (17, (4, 3)), (18, (4, 2)), (19, (4, 1)),
                   (20, (4, 0)), (21, (3, 0)), (22, (2, 0)), (23, (1, 0)), (24, (0, 0))]
    for i in range(Sxx.shape[-1]):  # Per channel
        ss = MinMaxScaler()
        scalars = Sxx[:, :, :, i].reshape(Sxx.shape[0], Sxx.shape[1]*Sxx.shape[2])
        scaled = ss.fit_transform(scalars)
        ch_pcs = PCA(scaled, n_components=n_components)
        # Spiral reshaping
        spiral_ch_pcs = np.zeros(shape=(ch_pcs.shape[0], 5, 5))
        for pc_idx in spiral_idxs:
            pc_num, idxs = pc_idx
            spiral_ch_pcs[:, idxs[0], idxs[1]] = ch_pcs[:, pc_num]

        features[:, :, i] = spiral_ch_pcs

    return features
