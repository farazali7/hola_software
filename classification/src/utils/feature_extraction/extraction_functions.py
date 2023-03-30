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


def feature_set_4(data, labels, args):
    """
    Compute spectogram EMG windows of time step x electrode channel x frequency.
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

    Sxx = np.transpose(Sxx, (0, 3, 2, 1))
    Sxx = Sxx[:, :, :, 1:]  # Ignore 0th frequency

    # Normalize to 0 to 1 range
    features = np.zeros_like(Sxx)
    for i in range(Sxx.shape[2]):  # Per channel
        ss = MinMaxScaler()
        scalars = Sxx[:, :, i, :].reshape(Sxx.shape[0], Sxx.shape[1]*Sxx.shape[3])
        scaled = ss.fit_transform(scalars)
        scaled = scaled.reshape(Sxx.shape[0], Sxx.shape[1], Sxx.shape[3])
        features[:, :, i, :] = scaled

    return features, homog_label_windows


def feature_set_5(data, labels, args):
    """
    Compute top PCs from spectrogram with Hamming window applied and reshape with most important PCs in middle.
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

    return features, homog_label_windows


def feature_set_6(data, labels, args):
    """
    Compute top PCs from spectrogram with Hamming window and reshape with most important PCs in middle, TRIAL-WISE
    :param data: Array of EMG data
    :param labels: Array of respective grasps labels to EMG data
    :param args: Dictionary of arguments such as window_size, window_overlap_size, etc.
    :return: Tuple of features and processed labels
    """
    trial_nums = np.unique(data[..., -1])

    all_features = []
    all_labels = []
    for trial_num in trial_nums:
        indices = np.where(data[..., -1] == trial_num)[0]
        trial_data = data[indices, :-1]
        trial_labels = labels[indices]

        trial_features, trial_labels_proc = feature_set_5(trial_data, trial_labels, args)

        trial_idx_arr = np.full(shape=(*trial_features.shape[:-1], 1), fill_value=trial_num)
        trial_features = np.concatenate([trial_features, trial_idx_arr], axis=-1)

        all_features.append(trial_features)
        all_labels.append(trial_labels_proc)

    features = np.vstack(all_features)
    labels = np.vstack(all_labels)

    return features, labels
