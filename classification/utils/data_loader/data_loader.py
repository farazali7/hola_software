import os
import pickle
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from classification.utils.signal_processing import window_data, homogenize_window
from classification.utils.feature_extraction.features import rms, mav, var, dwt


def save_formatted_data(data_path, subject_num, data_col, label_col, save_path):
    '''
    Helper function for saving formatted data to pickle file for easier future loading.
    :param data_path: String, path to main data directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data
    :param label_col: String, column name for labels
    :param save_path: String, path to save new formatted data file to
    :return:
    '''
    emg_data, grasp_labels = get_subject_emg_data_from_raw(data_path, subject_num, data_col, label_col)
    combined_data = np.concatenate([emg_data, grasp_labels], axis=1)
    with open(save_path, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done saving formatted data to : {save_path}')


def get_subject_emg_data_from_raw(data_path, subject_num, data_col, label_col):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data + labels for given subject from the raw
    dataset.
    :param data_path: String, path to main data directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data
    :param label_col: String, column name for labels
    :return Tuple of two NumPy arrays as (emg data, grasp labels)
    '''
    with open(os.path.join(data_path, subject_num.upper() + '_data.pkl'), 'rb') as handle:
        data = pickle.load(handle)

    # Get relevant labels only
    # 0 = hand relax, 5 = power sphere (TVG), 2 = lateral grasp (pinch)
    grasp_labels = data[label_col]
    rel_label_idxs = np.in1d(grasp_labels, [0, 2, 5])
    grasp_labels = grasp_labels[rel_label_idxs]

    # EMG data (electrodes 0,1,7 are near top and 3,5 are near bottom)
    # emg_data = data[data_col][rel_label_idxs][:, [0, 1, 3, 5, 7]]
    emg_data = data[data_col][rel_label_idxs]
    print(f'For {subject_num}: TVG labels: {np.bincount(np.squeeze(grasp_labels))[5]}, LP labels: {np.bincount(np.squeeze(grasp_labels))[2]}')

    return emg_data, grasp_labels


def get_subject_emg_data_from_proc(data_path, subject_num, data_col, label_col, labels_needed,
                                   emg_locs):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data + labels for given subject from the
    pre-formatted dataset.
    :param data_path: String, path to main data directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data
    :param label_col: String, column name for labels
    :param labels_needed: List of Int, specifying which grasp labels are needed
    :param emg_locs: List of Int, specifying which EMG electrodes from dataset to use as data sources
    :return Tuple of two NumPy arrays as (emg data, grasp labels)
    '''
    with open(os.path.join(data_path, subject_num.upper() + '_data.pkl'), 'rb') as handle:
        data = pickle.load(handle)

    # Get relevant labels only
    # 0 = hand relax, 5 = power sphere (TVG), 2 = lateral grasp (pinch)
    grasp_labels = data[:, -1].astype(np.int)
    rel_label_idxs = np.in1d(grasp_labels, labels_needed)
    grasp_labels = grasp_labels[rel_label_idxs]

    # EMG data (electrodes 0,1,7 are near top and 3,5 are near bottom)
    emg_data = data[rel_label_idxs][:, emg_locs]
    print(f'For {subject_num}: TVG labels: {np.bincount(np.squeeze(grasp_labels))[5]}, LP labels: {np.bincount(np.squeeze(grasp_labels))[2]}')

    return emg_data, grasp_labels


def get_subject_feature_data(data_path, subject_num, data_col, label_col, labels_needed, emg_locs):
    '''
    Retrieve features extracted for a given subject.
    :param data_path: String, path to main data directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data
    :param label_col: String, column name for labels
    :param labels_needed: List of Int, specifying which grasp labels are needed
    :param emg_locs: List of Int, specifying which EMG electrodes from dataset to use as data sources
    :return Tuple of two NumPy arrays as (features, grasp labels)
    '''
    emg_data, grasp_labels = get_subject_emg_data_from_proc(data_path, subject_num, data_col, label_col, labels_needed,
                                                            emg_locs)
    emg_windows = window_data(emg_data)
    grasp_labels_windows = window_data(grasp_labels)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

    # Compute features
    # RMS
    emg_rms = rms(emg_windows)

    # Mean Absolute Value (MAV)
    emg_mav = mav(emg_windows)

    # Variance
    emg_var = var(emg_windows)

    # marginal Discrete Wavelet Transform (mDWT)
    emg_mdwt = dwt(emg_windows, family='db7', level=3, axis=2)
    emg_mdwt_means = [np.nanmean(mdwt, 2) for mdwt in emg_mdwt]
    emg_mdwt_mean = np.concatenate(emg_mdwt_means)

    # Standardize features
    ss = StandardScaler()
    emg_rms_st = ss.fit_transform(emg_rms)
    emg_mav_st = ss.fit_transform(emg_mav)
    emg_var_st = ss.fit_transform(emg_var)
    emg_mdwt_mean_st = ss.fit_transform(emg_mdwt_mean)

    # Create full feature dataset
    full_emg_features = np.concatenate([emg_rms_st, emg_mav_st, emg_var_st, emg_mdwt_mean_st])
    full_labels = np.repeat(homog_label_windows, full_emg_features.shape[0] // homog_label_windows.shape[0])

    return full_emg_features, full_labels


def get_all_subject_data(data_path, subject_nums, data_col, label_col, labels_needed, emg_locs):
    '''
    Load and retrieve a list of emg features + labels for all subjects specified.
    :param data_path: String, path to main data directory
    :param subject_nums: List of String, all requested subject numbers as 's#' for subject #
    :param data_col: String, column name for data
    :param label_col: String, column name for labels
    :param labels_needed: List of Int, specifying which grasp labels are needed
    :param emg_locs: List of Int, specifying which EMG electrodes from dataset to use as data sources
    :return: Tuple of lists as (features, grasp labels)
    '''
    all_subjects_features = []
    all_subjects_labels = []
    for subject_num in subject_nums:
        subject_features, subject_labels = get_subject_feature_data(data_path, subject_num, data_col, label_col,
                                                                    labels_needed, emg_locs)
        all_subjects_features.append(subject_features)
        all_subjects_labels.append(subject_labels)

    return all_subjects_features, all_subjects_labels
