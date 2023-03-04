import os
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from classification.utils.signal_processing import window_data, homogenize_window
from classification.utils.feature_extraction.features import rms, mav, var, dwt
from classification.config import cfg


def format_ninaprodb10_data(data_path, subject_num, labels_needed, emg_locs):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_processing + labels for given subject from the
    pre-formatted dataset.
    :param data_path: String, path to main data_processing directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data_processing
    :param label_col: String, column name for labels
    :param labels_needed: List of Int, specifying which grasp labels are needed
    :param emg_locs: List of Int, specifying which EMG electrodes from dataset to use as data_processing sources
    :return Tuple of two NumPy arrays as (emg data_processing, grasp labels)
    '''
    data = loadmat(os.path.join(data_path, subject_num.upper() + '_data.pkl'))

    # Get static position only
    dynamic_state = data['dynamic']

    # Get relevant labels only
    # 0 = hand relax, 5 = power sphere (TVG), 2 = lateral grasp (pinch)
    grasp_labels = data[:, -1].astype(np.int32)
    rel_label_idxs = np.in1d(grasp_labels, labels_needed)
    grasp_labels = grasp_labels[rel_label_idxs]

    # EMG data_processing (electrodes 0,1,7 are near top and 3,5 are near bottom)
    emg_data = data[rel_label_idxs][:, emg_locs]
    print(f'For {subject_num}: TVG labels: {np.bincount(np.squeeze(grasp_labels))[5]}, LP labels: {np.bincount(np.squeeze(grasp_labels))[2]}')

    return emg_data, grasp_labels


def get_subject_feature_data(emg_data, grasp_labels, window_size, window_overlap_size,
                             combine_channels=False, standardize=False):
    '''
    Retrieve features extracted for a given subject.
    :param emg_data: Array of EMG data
    :param grasp_labels: Array of respective graps labels to EMG data
    :param window_size: Integer, number of samples in one window
    :param window_overlap_size: Integer, number of overlapping samples between windows
    :param combine_channels: Boolean, True to combine channels for features, False to keep separate
    :return Tuple of two NumPy arrays as (features, grasp labels)
    '''
    emg_windows = window_data(emg_data, window_size, window_overlap_size)
    grasp_labels_windows = window_data(grasp_labels, window_size, window_overlap_size)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

    # Compute features
    axis = combine_channels + 1

    # RMS
    emg_rms = rms(emg_windows, axis=axis)

    # Mean Absolute Value (MAV)
    emg_mav = mav(emg_windows, axis=axis)

    # Variance
    emg_var = var(emg_windows, axis=axis)

    # marginal Discrete Wavelet Transform (mDWT)
    emg_mdwt = dwt(emg_windows, family='db7', level=3, axis=axis)
    emg_mdwt_means = [np.nanmean(mdwt, axis) for mdwt in emg_mdwt]
    emg_mdwt_mean = np.concatenate(emg_mdwt_means)

    # Standardize features
    ss = StandardScaler() if standardize else MinMaxScaler()
    emg_rms_st = ss.fit_transform(emg_rms)
    emg_mav_st = ss.fit_transform(emg_mav)
    emg_var_st = ss.fit_transform(emg_var)
    emg_mdwt_mean_st = ss.fit_transform(emg_mdwt_mean)

    # Create full feature dataset
    full_emg_features = {'rms': emg_rms_st,
                         'mav': emg_mav_st,
                         'var': emg_var_st,
                         'mdwt': emg_mdwt_mean_st}
    # full_emg_features = np.concatenate([emg_rms_st, emg_mav_st, emg_var_st, emg_mdwt_mean_st])
    full_labels = {}
    for feature in full_emg_features.keys():
        full_labels[feature] = np.repeat(homog_label_windows, full_emg_features[feature].shape[0] // homog_label_windows.shape[0])

    return full_emg_features, full_labels


def get_all_subject_data(dataset_params, subject_nums):
    '''
    Load and retrieve a list of emg features + labels for all subjects specified.
    :param dataset_params: Dictionary of parameters to utilize for data_processing processing/feature extraction
    :param subject_nums: List of String formatted subject numbers
    :return: Tuple of lists as (features, grasp labels)
    '''
    all_subjects_features = []
    all_subjects_labels = []
    for subject_num in subject_nums:
        subject_features, subject_labels = get_subject_feature_data(**dataset_params, subject_num=subject_num)
        all_subjects_features.append(subject_features)
        all_subjects_labels.append(subject_labels)

    return all_subjects_features, all_subjects_labels

if __name__ == '__main__':
    data_path = cfg['DATA_PATH']
    healthy_subjects = cfg['HEALTHY_SUBJECTS']
    affected_subjects = cfg['AFFECTED_SUBJECTS']
    subject_nums = healthy_subjects + affected_subjects
    save_dir = cfg['SAVE_MODEL_PATH']
    data_col = cfg['DATA_COL_NAME']
    label_col = cfg['LABEL_COL_NAME']
    labels_needed = cfg['GRASP_LABELS']
    emg_locs = cfg['EMG_ELECTRODE_LOCS']
    num_folds = cfg['CV_FOLDS']
    window_size = cfg['WINDOW_SIZE']
    window_overlap_size = cfg['WINDOW_OVERLAP_SIZE']
    combine_channels = cfg['COMBINE_CHANNELS']
    standardize = cfg['STANDARDIZE']

    dataset_params = {'data_path': data_path,
                      'labels_needed': labels_needed,
                      'emg_locs': emg_locs,
                      'window_size': window_size,
                      'window_overlap_size': window_overlap_size,
                      'combine_channels': combine_channels,
                      'standardize': standardize}

    all_subject_emg, all_subject_labels = get_all_subject_data(dataset_params, subject_nums)

    data_dict = {}
    labels_dict = {}
    for i in range(len(subject_nums)):
        data_dict[subject_nums[i]] = all_subject_emg[i]
        labels_dict[subject_nums[i]] = all_subject_labels[i]

    with open(os.path.join(data_path, 'grasp_2_5_w400_sepch_norm_data.pkl'), 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_path, 'grasp_2_5_w400_sepch_norm_labels.pkl'), 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
