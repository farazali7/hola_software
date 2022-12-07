import datetime

import scipy
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pywt
import pickle


def window_data(data, window_size=200, overlap_size=100, remove_short=True, flatten_inside_window=False):
    '''
    Windowing function to split data based on window_size
    :param data: NumPy array of data
    :param window_size: Integer, number of samples in one window
    :param overlap_size: Integer, number of overlapping samples between windows
    :param remove_short: Boolean, set True to remove (last) shorter window
    :param flatten_inside_window: Boolean, set True to flatten window size dimension with outer dimension
    :return: Windowed array view
    '''
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, remove last shorter window
    if overhang != 0 and remove_short:
        data = data[:-overhang]

    sz = data.dtype.itemsize
    ret = as_strided(
            data,
            shape=(num_windows, window_size*data.shape[1]),
            strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def homogenize_window(window_data):
    '''
    Homogenize data in a window based on the mode (most frequently occurring value). Useful for ground-truth windows.
    :param window_data: Windowed data with >= 2 dims
    :return: New array containing mode across each row
    '''

    return scipy.stats.mode(np.squeeze(window_data), 1).mode


# Load all subjects' data
def get_subject_emg_data(data_path, subject_num):
    '''
    Load and retrieve NumPy arrays containing pertinent EMG classification data + labels for given subject.
    :param data_path: String, path to main data directory
    :param subject_num: String, subject number specified as 's#' for subject #
    :return Tuple of two NumPy arrays as (emg data, grasp labels)
    '''
    mat_path = os.path.join(data_path, subject_num, subject_num.upper() + '_E3_A1.mat')
    data = scipy.io.loadmat(mat_path)

    # Get relevant labels only
    # 0 = hand relax, 5 = medium wrap (TVG), 17 = lateral grasp (pinch)
    grasp_labels = data['restimulus']
    rel_label_idxs = np.in1d(grasp_labels, [5, 17])
    grasp_labels = grasp_labels[rel_label_idxs]

    # EMG data
    emg_data = data['emg'][rel_label_idxs, :8]
    print(f'For {subject_num}: TVG labels: {np.bincount(np.squeeze(grasp_labels))[5]}, LP labels: {np.bincount(np.squeeze(grasp_labels))[17]}')
    emg_windows = window_data(emg_data)
    grasp_labels_windows = window_data(grasp_labels)

    # Assign grasp label windows to mode label
    homog_label_windows = homogenize_window(grasp_labels_windows)

    # Compute features
    # RMS
    emg_rms = np.sqrt(np.mean(emg_windows**2, 2))

    # Mean Absolute Value (MAV)
    emg_mav = np.mean(np.abs(emg_windows), 2)

    # Variance
    emg_var = np.var(emg_windows, 2)

    # marginal Discrete Wavelet Transform (mDWT) ?
    emg_mdwt = pywt.wavedec(emg_windows, 'db7', level=3, axis=2)
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
    full_labels = np.repeat(homog_label_windows, full_emg_features.shape[0]//homog_label_windows.shape[0])

    return full_emg_features, full_labels


def get_all_subject_data(data_path, subject_nums):
    '''
    Load and retrieve a list of emg data + labels for all subjects specified.
    :param data_path: String, path to main data directory
    :param subject_nums: List of String, all requested subject numbers as 's#' for subject #
    :return: Tuple of lists as (emg data, grasp labels)
    '''
    all_subjects_emg = []
    all_subjects_labels = []
    for subject_num in subject_nums:
        subject_emg, subject_labels = get_subject_emg_data(data_path, subject_num)
        all_subjects_emg.append(subject_emg)
        all_subjects_labels.append(subject_labels)

    return all_subjects_emg, all_subjects_labels


def LOSO_CV(data_path, subject_nums, model, save_dir):
    '''
    Perform Leave-One-Subject-Out Cross-Validation for given dataset and subjects.
    :param data_path: String, path to main data directory.
    :param subject_nums: List of String, subject numbers requested in CV as 's#' for subject #
    :param model: Classification model to train and validate (ex. kNN, SVM, DT, etc.)
    :param save_dir: String, path to directory where results should be saved
    '''

    all_subject_emg, all_subject_labels = get_all_subject_data(data_path, subject_nums)

    main_save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Create results directory if not already made
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    for idx in range(len(all_subject_emg)):
        print(f'\n----- PERFORMING LOSO-CV FOR SUBJECT: {idx+1} -----\n')
        # Create training set
        train_data = all_subject_emg[:idx] + all_subject_emg[idx+1:]
        train_data = np.concatenate(train_data, 0)
        train_labels = all_subject_labels[:idx] + all_subject_labels[idx + 1:]
        train_labels = np.concatenate(train_labels, 0)

        # Create validation set
        val_data = all_subject_emg[idx]
        val_labels = all_subject_labels[idx]

        # Fit to model
        model.fit(train_data, train_labels)

        # Predict with model
        preds = model.predict(val_data)

        # Save model, preds, and val labels
        np.savetxt(os.path.join(main_save_dir, f'cv{idx+1}_preds.csv'), preds, delimiter=',')
        np.savetxt(os.path.join(main_save_dir, f'cv{idx+1}_val_labels.csv'), val_labels, delimiter=',')
        pickle.dump(model, open(os.path.join(main_save_dir, f'cv{idx+1}_model.pkl'), 'wb'))

        print('Done.')


if __name__ == "__main__":
    data_path = 'data/'
    subject_nums = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    save_dir = 'results/'
    knn_clf = KNeighborsClassifier(n_jobs=-1)
    svm_clf = SVC(verbose=1)
    dt_clf = DecisionTreeClassifier()
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 10))
    models = {'KNN': knn_clf,
              'SVM': svm_clf,
              'DecisionTree': dt_clf,
              'MLP': mlp_clf}

    for model_name in models.keys():
        print(f'VALIDATING MODEL: {model_name}')
        model = models[model_name]
        save_dir_model = os.path.join(save_dir, model_name)
        LOSO_CV(data_path, subject_nums, model, save_dir_model)

    print('done')