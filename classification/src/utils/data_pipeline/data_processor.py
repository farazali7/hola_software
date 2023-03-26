import os
import numpy as np
from classification.src.utils.preprocessing import butter_bandpass_filter, notch_filter, \
    downsample
from classification.src.utils.data_pipeline import save_data, load_data
from classification.src.config import cfg
from classification.src.constants import FEATURE_EXTRACTION_FUNCS

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def preprocess_data(emg_data, grasp_labels, butter_ord, butter_freq, notch_freq, qf, sampling_freq, target_freq=None,
                    from_np=False, save_path=None):
    """
    Preprocess the data by filtering and restructuring class labels etc.
    :param emg_data: Array of EMG data
    :param grasp_labels: Array of respective grasp labels to EMG data
    :param butter_ord: Integer for butterworth filter order
    :param butter_freq: Cutoff frequency(ies)
    :param notch_freq: Frequency to remove from data
    :param qf: Quality factor, represents the ratio between center frequency and bandwidth (high for notch)
    :param sampling_freq: Integer for sampling frequency of emg_data
    :param target_freq: Integer for desired sampling frequency if specified
    :param from_np: Boolean denoting if data from Ninapro is being used (if True, this will regroup grasp labels)
    :param save_path: String, path to save the data in
    """
    # Remove cable sway and extraneous frequencies
    emg_data = butter_bandpass_filter(emg_data, butter_ord, butter_freq, sampling_freq)

    # Remove main line interference
    emg_data = notch_filter(emg_data, notch_freq, qf, sampling_freq)

    # Resample if desired
    if target_freq is not None:
        data = np.concatenate([emg_data, grasp_labels], axis=1)
        resampled_data = downsample(data, target_freq, sampling_freq)
        last_dim = resampled_data.shape[1] - 1
        emg_data, grasp_labels = resampled_data[:, :last_dim], resampled_data[:, last_dim:]

    # Assign non-TVG, non-LP labels to rest class (i.e., make a 'catch-all' class)
    if from_np:
        grasp_labels = np.where(grasp_labels > 2, 0, grasp_labels)

    grasp_labels = grasp_labels.astype(int)

    if save_path:
        save_data(emg_data, grasp_labels, save_path)

    return emg_data, grasp_labels


def extract_features(data, labels, feature_extraction_func, feature_extraction_args, save_path=None):
    '''
    Retrieve features extracted for a given data array.

    :param save_path: String, path to save the data in
    :return: Tuple of two NumPy arrays as (features, grasp labels)
    '''

    emg_features, labels = feature_extraction_func(data, labels, feature_extraction_args)

    if save_path:
        save_data(emg_features, labels, save_path)

    return emg_features, labels


def process_data(subject_id, data_dir, preprocessing_args, feature_extraction_func,
                 feature_extraction_args, save_dir=None):
    """
    Process a subject's EMG data by preprocessing (filtering) and extracting features (+windowing).
    :param subject_id: String, subject ID specified by dataset file naming for specific subject
    :param data_dir: String, path to main dataset directory
    :param preprocessing_args: Dictionary of keyword args for preprocessing
    :param feature_extraction_func: Function for feature extraction
    :param feature_extraction_args: Dictionary of keyword args for feature extraction
    :param save_dir: String, path to directory to save the data in
    :return: Tuple of processed data and labels
    """
    data_path = os.path.join(data_dir, subject_id + '.pkl')
    data, labels = load_data(data_path)

    # Preprocessing
    preprocessed_data, preprocessed_labels = preprocess_data(data, labels, **preprocessing_args)

    # Feature extraction
    features, feature_labels = extract_features(preprocessed_data, preprocessed_labels,
                                                feature_extraction_func, feature_extraction_args)

    if save_dir:
        save_path = os.path.join(save_dir, subject_id + '.pkl')
        save_data(features, feature_labels, save_path)

    return features, feature_labels


if __name__ == '__main__':
    # Preprocessing args
    butter_ord = cfg['BUTTERWORTH_ORDER']
    butter_freq = cfg['BUTTERWORTH_FREQ']
    notch_freq = cfg['NOTCH_FREQ']
    qf = cfg['QUALITY_FACTOR']
    target_freq = cfg['TARGET_FREQ']

    # Feature extraction args
    window_size = cfg['WINDOW_SIZE']
    window_overlap_size = cfg['WINDOW_OVERLAP_SIZE']
    combine_channels = cfg['COMBINE_CHANNELS']
    standardize = cfg['STANDARDIZE']
    feature_extraction_func = FEATURE_EXTRACTION_FUNCS[cfg['FEATURE_EXTRACTION_FUNC']]

    # NINAPRO DB10
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    formatted_data_path = np_cfg['FORMATTED_DATA_PATH']
    healthy_subjects = np_cfg['HEALTHY_SUBJECTS']
    affected_subjects = np_cfg['AFFECTED_SUBJECTS']
    subject_ids = healthy_subjects + affected_subjects
    save_dir = np_cfg['PROCESSED_DATA_PATH']
    np_sampling_freq = np_cfg['SAMPLING_FREQ']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocessing_args = {'butter_ord': butter_ord,
                          'butter_freq': butter_freq,
                          'notch_freq': notch_freq,
                          'qf': qf,
                          'sampling_freq': np_sampling_freq,
                          'target_freq': target_freq,
                          'from_np': True}

    feature_extraction_args = {'window_size': window_size,
                               'window_overlap_size': window_overlap_size,
                               'combine_channels': combine_channels,
                               'standardize': standardize,
                               'sampling_freq': target_freq}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(process_data,
                                          data_dir=formatted_data_path,
                                          preprocessing_args=preprocessing_args,
                                          feature_extraction_func=feature_extraction_func,
                                          feature_extraction_args=feature_extraction_args,
                                          save_dir=save_dir), subject_ids),
                        total=len(subject_ids)))

    print('Done.')

    # GrabMyo
    gm_cfg = cfg['DATASETS']['GRABMYO']
    formatted_data_path = gm_cfg['FORMATTED_DATA_PATH']
    healthy_subjects = gm_cfg['HEALTHY_SUBJECTS']
    subject_ids = ['S' + str(x + 115) for x in healthy_subjects]
    save_dir = gm_cfg['PROCESSED_DATA_PATH']
    gm_sampling_freq = gm_cfg['SAMPLING_FREQ']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocessing_args = {'butter_ord': butter_ord,
                          'butter_freq': butter_freq,
                          'notch_freq': notch_freq,
                          'qf': qf,
                          'sampling_freq': gm_sampling_freq,
                          'target_freq': target_freq,
                          'from_np': False}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(process_data,
                                          data_dir=formatted_data_path,
                                          preprocessing_args=preprocessing_args,
                                          feature_extraction_func=feature_extraction_func,
                                          feature_extraction_args=feature_extraction_args,
                                          save_dir=save_dir), subject_ids),
                        total=len(subject_ids)))

    print('Done.')
