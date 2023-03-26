from classification.src.utils.preprocessing import window_data, homogenize_window
from classification.src.utils.feature_extraction.features import rms, mav, var, dwt, hjorth_complexity, spectogram, PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import numpy as np


def feature_set_1(emg_data, grasp_labels, args):
    """
    Compute RMS, MAV, VAR, mDWT, Mobility, and Complexity.
    :param emg_data: Array of EMG data
    :param grasp_labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    window_size = args['window_size']
    window_overlap_size = args['window_overlap_size']
    combine_channels = args['combine_channels']
    standardize = args['standardize']

    emg_windows = window_data(emg_data, window_size, window_overlap_size)
    grasp_labels_windows = window_data(grasp_labels, window_size, window_overlap_size)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

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
    labels = np.repeat(homog_label_windows, emg_features.shape[0] // homog_label_windows.shape[0])[..., np.newaxis]

    return emg_features, labels


def feature_set_2(data, labels, args):
    """
    Compute top PCs from spectrogram with Hamming window applied.
    :param data: Array of EMG data
    :param labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    window_size = args['window_size']
    window_overlap_size = args['window_overlap_size']
    sampling_freq = args['sampling_freq']

    emg_windows = window_data(data, window_size, window_overlap_size)
    grasp_labels_windows = window_data(labels, window_size, window_overlap_size)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

    # Apply hamming window and get spectogram
    window = 'hamming'
    f, t, Sxx = spectogram(data=emg_windows, sampling_freq=sampling_freq, window=window, nperseg=20,
                           n_overlap=10, axis=1)

    Sxx = np.transpose(Sxx, (0, 3, 1, 2))

    # Normalize to 0 to 1 range & PCA
    n_components = 25
    features = np.zeros(shape=(Sxx.shape[0], n_components, Sxx.shape[-1]))
    for i in range(Sxx.shape[-1]):  # Per channel
        ss = MinMaxScaler()
        scalars = Sxx[:, :, :, i].reshape(Sxx.shape[0], Sxx.shape[1]*Sxx.shape[2])
        scaled = ss.fit_transform(scalars)
        ch_pcs = PCA(scaled, n_components=n_components)
        features[:, :, i] = ch_pcs

    return features, homog_label_windows


def feature_set_3(data, labels, args):
    """
    Compute raw EMG windows.
    :param data: Array of EMG data
    :param labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    window_size = args['window_size']
    window_overlap_size = args['window_overlap_size']

    emg_windows = window_data(data, window_size, window_overlap_size)
    grasp_labels_windows = window_data(labels, window_size, window_overlap_size)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

    # Normalize each channel
    ss = MinMaxScaler()
    features = np.zeros_like(emg_windows)
    for i in range(emg_windows.shape[0]):
        window = emg_windows[i, :, :]
        norm_window = ss.fit_transform(window)
        features[i, :, :] = norm_window

    return features, homog_label_windows
