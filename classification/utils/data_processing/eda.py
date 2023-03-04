import random

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from classification.config import cfg


def plot_signal(signal_data, time_vec=None, x_label='', y_label='', title=''):
    """
    Plot signal data.
    :param signal_data: Array of signal data
    :param time_vec: Array of time vector or if None defaults to no. of samples in signal_data
    :param x_label: String for x-axis label
    :param y_label: String for y-axis label
    :param title: String for plot title
    """
    if len(signal_data.shape) > 1:
        signal_data = np.squeeze(signal_data)
    if not time_vec:
        time_vec = np.arange(signal_data.shape[0])
    elif len(time_vec.shape) > 1:
        time_vec = np.squeeze(time_vec)

    plt.plot(time_vec, signal_data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def plot_batch_signal(signal_data, time_vec=None, x_label='', y_label='', title_base='', n_subjects=6):
    """
    Plot a batch of subjects' signal data
    :param signal_data: Dict of arrays with signal data per subject as keys
    :param time_vec: List of arrays of time vectors or if None defaults to no. of samples per signal_data
    :param x_label: String for x-axis label
    :param y_label: String for y-axis label
    :param title_base: String for plot title base (will be appended by subject info per plot)
    :param n_subjects: Int for number of subjects to display data from
    """
    # Find rows and cols
    rows = int(np.floor(np.sqrt(n_subjects)))
    cols = n_subjects // rows
    fig, axs = plt.subplots(rows, cols, figsize=[9.6, 7.2])

    samples_on_x = time_vec is None
    for i, key in enumerate(signal_data.keys()):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if samples_on_x:
            time_vec = np.arange(signal_data[key].shape[0])
        ax.plot(time_vec, signal_data[key])
        ax.set(xlabel=x_label, ylabel=y_label, title=key)
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout(pad=0.8)
    fig.suptitle(title_base)
    plt.subplots_adjust(top=0.87, hspace=0.4)
    plt.show()


def load_data(data_path):
    """
    Load emg and labels from pickle file.
    :param data_path: String for path to .pkl data file
    :return: Tuple of emg data, labels
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data[:, :-1], data[:, -1:]


def segment_by_grasp(emg_data, labels, grasp_id):
    """
    Return subset of EMG data based on position idxs of desired grasp
    :param emg_data: Array of emg data
    :param labels: Array of graps labels
    :param grasp_id: Int representing ID of desired grasp
    :returns: Array of emg data corresponding to desired grasp
    """
    grasp_idxs = np.where(labels == grasp_id)[0]
    return emg_data[grasp_idxs, :]


def get_batch_grasp_data(dir_path, grasp_id, subject_ids, channels=None, n_subjects=6):
    """
    Get a batch of n_subjects' EMG data pertaining to desired grasp
    :param dir_path: String path to data directory
    :param grasp_id: Int representing ID of desired grasp
    :param subject_ids: List of string IDs for subjects to randomly sample from
    :param channels: Electrode indices to take, if None all are returned
    :param n_subjects: Int for number of subjects to display data from
    :return: Dictionary of relevant EMG grasp data of subjects
    """
    random.seed(7)
    subjects_to_retrieve = random.sample(subject_ids, n_subjects)
    batch_data = {}

    for id in subjects_to_retrieve:
        file_path = os.path.join(dir_path, id + '.pkl')
        emg, labels = load_data(file_path)
        grasp_emg = segment_by_grasp(emg, labels, grasp_id)
        if channels is not None:
            grasp_emg = grasp_emg[:, channels]
        batch_data[id] = grasp_emg

    return batch_data


if __name__ == '__main__':
    CHANNELS = [0]  # Just visualize one channel for now

    # Load samples from NinaPro DB10
    data_path = 'data/formatted/ninapro_db10/'
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    np_ids = np_cfg['HEALTHY_SUBJECTS'] + np_cfg['AFFECTED_SUBJECTS']

    # Get Rest data
    np_rest_data = get_batch_grasp_data(data_path, 0, np_ids, 0)
    # Get TVG data
    np_tvg_data = get_batch_grasp_data(data_path, 1, np_ids, 0)
    # Get LP data
    np_lp_data = get_batch_grasp_data(data_path, 2, np_ids, 0)

    time_vec = None  # Will plot just samples for now

    plot_batch_signal(np_rest_data, time_vec, x_label='Samples', y_label='EMG', title_base='NinaPro DB10 Rest')
    plot_batch_signal(np_tvg_data, time_vec, x_label='Samples', y_label='EMG', title_base='NinaPro DB10 TVG')
    plot_batch_signal(np_lp_data, time_vec, x_label='Samples', y_label='EMG', title_base='NinaPro DB10 LP')

    # Load samples from GrabMyo
    data_path = 'data/formatted/grabmyo_openhand/'
    gm_cfg = cfg['DATASETS']['GRABMYO']
    gm_ids = gm_cfg['HEALTHY_SUBJECTS']
    gm_ids_fmt = ['S'+str(i+115) for i in gm_ids]  # Reformat int ids to match NinaPro

    # Get OH data
    gm_oh_data = get_batch_grasp_data(data_path, -1, gm_ids_fmt, 0)

    time_vec = None  # Will plot just samples for now

    plot_batch_signal(gm_oh_data, time_vec, x_label='Samples', y_label='EMG', title_base='GrabMyo OH')