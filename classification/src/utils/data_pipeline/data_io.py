import numpy as np
import pickle
import os
from tqdm import tqdm


def save_data(emg_data, grasp_labels, save_path):
    '''
    Helper function for saving formatted data to pickle file for easier future loading.
    :param data_path: String, path to main data_pipeline directory
    :param subject_id: String, subject number specified by database file naming for specific subject
    :param data_col: String, column name for data_pipeline
    :param label_col: String, column name for labels
    :param save_path: String, path to save new formatted data_pipeline file to
    :return:
    '''
    combined_data = (emg_data, grasp_labels)
    with open(save_path, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(data_path):
    """
    Load emg and labels from pickle file.
    :param data_path: String for path to .pkl data file
    :return: Tuple of emg data, labels
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, tuple):
        x, y = data[0], data[1]
    else:
        x, y = data[:, :-1], data[:, -1:]
    return x, y


def convert_to_full_paths(file_names, base_path):
    """
    Create full path to files by prepending with a given base path.
    :param file_names: List of file names
    :param base_path: String specifying base path
    :return: List of full file paths
    """
    return [os.path.join(base_path, file_name) for file_name in file_names]


def load_and_concat(file_names, ext=None):
    """
    Load and combine data (X and y) from multiple files. Add given extension if present to each file before loading.
    :param file_names: List of file names
    :param ext: String specifying file extension to append to each file name if given
    :return: Tuple of Numpy arrays as (X, y)
    """
    all_x = []
    all_y = []
    for file in tqdm(file_names, total=len(file_names)):
        path = file + (ext if ext is not None else '')
        X, y = load_data(path)
        all_x.append(X)
        all_y.append(y)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y).astype(np.int8)

    return all_x, all_y
