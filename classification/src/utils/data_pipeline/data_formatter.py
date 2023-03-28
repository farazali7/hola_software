import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import wfdb
from scipy.io import loadmat
from tqdm import tqdm

from classification.src.config import cfg
from classification.src.utils.data_pipeline import save_data


def format_ninapro_db10_data(subject_id, data_path, data_col, label_col, grasp_ids, electrode_ids, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_pipeline + labels for given subject from the raw
    dataset.
    :param data_path: String, path to main dataset directory
    :param subject_id: String, subject number specified by dataset file naming for specific subject
    :param data_col: String, column name for data_pipeline
    :param label_col: String, column name for labels,
    :param grasp_ids: List of grasp int IDs
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_pipeline, grasp labels)
    '''
    data = loadmat(os.path.join(data_path, subject_id.upper() + '_ex1.mat'))

    # Get movement done in static condition
    dynamic_state = data['redynamic']
    static_idxs = np.isin(dynamic_state, 0).ravel()

    # Remove unreliable data as stated by Cognolato et. al 2020 (Ninapro DB10 paper)
    bad_rep = data['reobjectrepetition']
    bad_rep_idxs = np.isin(bad_rep, [-2, -3]).ravel()
    good_rep_idxs = np.logical_not(bad_rep_idxs)

    # Get grasp label indices
    grasp_idxs = np.isin(data[label_col], grasp_ids).ravel()
    filter_idxs = np.logical_and(static_idxs, grasp_idxs)
    filter_idxs = np.logical_and(filter_idxs, good_rep_idxs)

    # Apply filters and get relevant electrode channels
    emg_data = data[data_col][filter_idxs][:, electrode_ids]
    grasp_labels = data[label_col][filter_idxs]

    if save_dir:
        save_path = os.path.join(save_dir, subject_id + '.pkl')
        save_data(emg_data, grasp_labels, save_path)

    return emg_data, grasp_labels


def format_grabmyo_data(subject_id, all_records, electrode_ids, grasp_ids, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_pipeline + labels for given subject from the raw
    dataset.
    :param subject_id: Int, subject number specified by database file naming for specific subject
    :param all_records: Dictionary specifying PhysioNet records for subjects (not unique here for multiprocessing)
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param new_label: Int specifying what number to assign to open hand labels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_pipeline, grasp labels)
    '''
    subject_records = all_records[subject_id]
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
    grasp_labels = np.full(bipolar_data.shape[0], grasp_ids['OH'])[..., np.newaxis]

    subject_id += 115  # Offset from subjects in NinaProDB10

    if save_dir:
        save_path = os.path.join(save_dir, 'S' + str(subject_id) + '.pkl')
        save_data(bipolar_data, grasp_labels, save_path)

    return bipolar_data, grasp_labels

def format_grabmyo_data_all(subject_id, all_records, electrode_ids, grasp_ids, save_dir=None):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data_pipeline + labels for given subject from the raw
    dataset.
    :param subject_id: Int, subject number specified by database file naming for specific subject
    :param all_records: Dictionary specifying PhysioNet records for subjects (not unique here for multiprocessing)
    :param electrode_ids: Array of ints representing IDs for electrode channels
    :param grasp_ids: Dict of int specifying what number to assign to new labels
    :param save_dir: String, path to directory to save the data in
    :return Tuple of two NumPy arrays as (emg data_pipeline, grasp labels)
    '''
    subject_records = all_records[subject_id]
    data_oh = []
    data_tvg = []
    data_lp = []
    for record in subject_records:
        folder, record_file = record.rsplit('/', 1)
        signal, md = wfdb.rdsamp(record_file, pn_dir='grabmyo/' + folder)
        if "gesture1_" in record:
            data_lp.append(signal)
        elif "gesture15" in record:
            data_oh.append(signal)
        elif "gesture16" in record:
            data_tvg.append(signal)

    all_data_lists = [data_oh, data_tvg, data_lp]
    all_data_arrs = [np.array(d) for d in all_data_lists]
    all_data = [np.reshape(d, (-1, d.shape[-1])) for d in all_data_arrs]

    # Convert monopolar electrodes to bipolar by subtracting
    bipolar_grasps = []
    for grasp_data in all_data:
        bipolar_data = []
        for channel_pair in electrode_ids:
            channels = grasp_data[:, channel_pair]
            differential = channels[:, 0] - channels[:, 1]  #
            bipolar_data.append(differential)
        bipolar_grasps.append(bipolar_data)

    bipolar_grasps = np.hstack(bipolar_grasps)
    oh_labels = np.full(shape=(1, bipolar_grasps.shape[-1]//3), fill_value=grasp_ids['OH'])
    tvg_labels = np.full(shape=(1, bipolar_grasps.shape[-1]//3), fill_value=grasp_ids['TVG'])
    lp_labels = np.full(shape=(1, bipolar_grasps.shape[-1]//3), fill_value=grasp_ids['LP'])
    labels = np.hstack([oh_labels, tvg_labels, lp_labels])

    bipolar_data = np.transpose(np.array(bipolar_grasps))
    grasp_labels = np.transpose(np.array(labels))

    subject_id += 115  # Offset from subjects in NinaProDB10

    if save_dir:
        save_path = os.path.join(save_dir, 'S' + str(subject_id) + '.pkl')
        save_data(bipolar_data, grasp_labels, save_path)

    return bipolar_data, grasp_labels


def conditions_met(record):
    '''
    Make a series of conditions on a record from PhysioNet database to filter list
    :record: String containing path to record
    :returns: Boolean indicating if record meets all conditions
    '''
    sessions = ['session1']
    oh = 'gesture15'
    tvg = 'gesture16'
    lp = 'gesture1_'
    gestures = [oh]
    return any(session in record for session in sessions) and any(gesture in record for gesture in gestures)


def organize_pn_records(record_list):
    """
    Helper function to organize record list from a PhysioNet dataset by subject
    :param record_list: Set or List of record strings
    :return: Dictionary of records with subjects as keys
    """
    recs_by_subject = {}
    for record in record_list:
        subject_id = int(record.split('/')[1].split('participant')[-1])
        if subject_id not in recs_by_subject.keys():
            recs_by_subject[subject_id] = []
        recs_by_subject[subject_id].append(record)

    return recs_by_subject


if __name__ == '__main__':
    # NINAPRO DB10
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    raw_data_path = np_cfg['RAW_DATA_PATH']
    healthy_subjects = np_cfg['HEALTHY_SUBJECTS']
    affected_subjects = np_cfg['AFFECTED_SUBJECTS']
    subject_ids = healthy_subjects + affected_subjects
    data_col = np_cfg['DATA_COL_NAME']
    label_col = np_cfg['LABEL_COL_NAME']
    grasp_ids = np_cfg['GRASP_LABELS']
    electrode_ids = np_cfg['ELECTRODE_IDS']
    save_dir = np_cfg['FORMATTED_DATA_PATH']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    format_params = {'data_path': raw_data_path,
                     'data_col': data_col,
                     'label_col': label_col,
                     'grasp_ids': grasp_ids,
                     'electrode_ids': electrode_ids,
                     'save_dir': save_dir}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(format_ninapro_db10_data, **format_params), subject_ids),
                        total=len(subject_ids)))

    print('Done.')

    # GRABMYO
    gm_cfg = cfg['DATASETS']['GRABMYO']
    electrode_ids = gm_cfg['ELECTRODE_IDS']
    healthy_subjects = gm_cfg['HEALTHY_SUBJECTS']
    subject_ids = healthy_subjects
    grasp_ids = gm_cfg['GRASP_IDS']
    save_dir = gm_cfg['FORMATTED_DATA_PATH']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    records = wfdb.get_record_list('grabmyo')
    filtered_records = set([record.split('\n')[0] for record in records if conditions_met(record)])
    records_by_subject = organize_pn_records(filtered_records)

    format_params = {'electrode_ids': electrode_ids,
                     'all_records': records_by_subject,
                     'grasp_ids': grasp_ids,
                     'save_dir': save_dir}

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(format_grabmyo_data, **format_params), records_by_subject.keys()),
                        total=len(records_by_subject.keys())))

    print('Done.')
