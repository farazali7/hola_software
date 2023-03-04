import os
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from classification.utils.signal_processing import window_data, homogenize_window
from classification.utils.feature_extraction.features import rms, mav, var, dwt
from classification.config import cfg
import wfdb
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def save_data(emg_data, grasp_labels, save_path):
    '''
    Helper function for saving formatted data to pickle file for easier future loading.
    :param data_path: String, path to main data_processing directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data_processing
    :param label_col: String, column name for labels
    :param save_path: String, path to save new formatted data_processing file to
    :return:
    '''
    combined_data = np.concatenate([emg_data, grasp_labels], axis=1)
    with open(save_path, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def format_ninapro_db10_data(subject_num, data_path, data_col, label_col, electrode_ids, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_processing + labels for given subject from the raw
    dataset.
    :param data_path: String, path to main data_processing directory
    :param subject_num: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data_processing
    :param label_col: String, column name for labels,
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_processing, grasp labels)
    '''
    data = loadmat(os.path.join(data_path, subject_num.upper() + '_ex1.mat'))

    # Get movement done in static condition
    dynamic_state = data['redynamic']
    static_idxs = np.isin(dynamic_state, 0).ravel()

    # filter_idxs = np.logical_and(static_idxs)
    filter_idxs = static_idxs

    # Apply filters and get relevant electrode channels
    emg_data = data[data_col][filter_idxs][:, electrode_ids]
    grasp_labels = data[label_col][filter_idxs]

    if save_dir:
        save_path = os.path.join(save_dir, subject_num + '.pkl')
        save_data(emg_data, grasp_labels, save_path)

    return emg_data, grasp_labels


def format_grabmyo_data(subject_num, all_records, electrode_ids, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_processing + labels for given subject from the raw
    dataset.
    :param subject_num: Int, subject number specified by database file naming for specific subject
    :param all_records: Dictionary specifying PhysioNet records for subjects (not unique here for multiprocessing)
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_processing, grasp labels)
    '''
    subject_records = all_records[subject_num]
    data = []
    for record in subject_records:
        folder, record_file = record.rsplit('/', 1)
        signal, md = wfdb.rdsamp(record_file, pn_dir='grabmyo/' + folder)
        data.append(signal)

    data = np.array(data)
    data = data.reshape(-1, data.shape[-1])

    # Convert monopolar electrodes to bipolar by subtracting
    bipolar_data = []
    for channel_pair in electrode_ids:
        channels = data[:, channel_pair]
        differential = channels[:, 0] - channels[:, 1]  #
        bipolar_data.append(differential)
    bipolar_data = np.transpose(np.array(bipolar_data))
    grasp_labels = np.full_like(bipolar_data, -1)

    subject_num += 115  # Offset from subjects in NinaProDB10

    if save_dir:
        save_path = os.path.join(save_dir, 'S' + str(subject_num) + '.pkl')
        save_data(bipolar_data, grasp_labels, save_path)

    return bipolar_data, grasp_labels


def conditions_met(record):
    '''
    Make a series of conditions on a record from PhysioNet database to filter list
    :record: String containing path to recrod
    :returns: Boolean indicating if record meets all conditions
    '''
    session = 'session1'
    gesture = 'gesture15'  # Open hand
    return session in record and gesture in record


def organize_pn_records(record_list):
    """
    Helper function to organize record list from a PhysioNet dataset by subject
    :param record_list: Set or List of record strings
    :return: Dictionary of records with subjects as keys
    """
    recs_by_subject = {}
    for record in record_list:
        subject_num = int(record.split('/')[1].split('participant')[-1])
        if subject_num not in recs_by_subject.keys():
            recs_by_subject[subject_num] = []
        recs_by_subject[subject_num].append(record)

    return recs_by_subject


if __name__ == '__main__':
    base_save_dir = 'data/formatted'

    # NINAPRO DB10
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    raw_data_path = np_cfg['RAW_DATA_PATH']
    healthy_subjects = np_cfg['HEALTHY_SUBJECTS']
    affected_subjects = np_cfg['AFFECTED_SUBJECTS']
    subject_nums = healthy_subjects + affected_subjects
    data_col = np_cfg['DATA_COL_NAME']
    label_col = np_cfg['LABEL_COL_NAME']
    electrode_ids = np_cfg['ELECTRODE_IDS']
    save_dir = os.path.join(base_save_dir, 'ninapro_db10')

    format_params = {'data_path': raw_data_path,
                     'data_col': data_col,
                     'label_col': label_col,
                     'electrode_ids': electrode_ids,
                     'save_dir': save_dir}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(format_ninapro_db10_data, **format_params), subject_nums),
                        total=len(subject_nums)))

    print('Done.')

    # GRABMYO
    gm_cfg = cfg['DATASETS']['GRABMYO']
    electrode_ids = gm_cfg['ELECTRODE_IDS']
    healthy_subjects = gm_cfg['HEALTHY_SUBJECTS']
    subject_nums = healthy_subjects
    save_dir = os.path.join(base_save_dir, 'grabmyo_openhand')

    records = wfdb.get_record_list('grabmyo')
    filtered_records = set([record.split('\n')[0] for record in records if conditions_met(record)])
    records_by_subject = organize_pn_records(filtered_records)

    format_params = {'electrode_ids': electrode_ids,
                     'all_records': records_by_subject,
                     'save_dir': save_dir}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(format_grabmyo_data, **format_params), records_by_subject.keys()),
                        total=len(records_by_subject.keys())))

    print('Done.')
