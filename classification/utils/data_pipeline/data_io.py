import numpy as np
import pickle


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
    combined_data = np.concatenate([emg_data, grasp_labels], axis=1)
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

    return data[:, :-1], data[:, -1:]
