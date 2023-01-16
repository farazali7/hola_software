import datetime
import os
import pickle
import random

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classification.utils.data_loader import get_all_subject_data, save_formatted_data
from classification.config import cfg


def create_cv_folds(healthy_ids, affected_ids, num_folds):
    '''
    Create list of cross-validation folds containing equal proportion healthy-affected subject ids.
    :param healthy_ids: List of String, ids of healthy subjects
    :param affected_ids: List of String, ids of affected subjects
    :param num_folds: Int, represents number of folds
    '''
    rand = random.Random(4)
    num_healthy = len(healthy_ids)
    num_affected = len(affected_ids)
    affected_in_fold = np.floor(num_affected / num_folds).astype(np.int32)
    healthy_in_fold = np.floor(num_healthy / num_folds).astype(np.int32)
    healthy_in_fold = healthy_in_fold + 1 if healthy_in_fold == 0 else healthy_in_fold

    rand.shuffle(healthy_ids)
    rand.shuffle(affected_ids)

    healthy_id_subsets = [healthy_ids[i:i+healthy_in_fold] for i in range(0, num_healthy, healthy_in_fold)]
    affected_id_subsets = [affected_ids[i:i+affected_in_fold] for i in range(0, num_affected, affected_in_fold)]

    # Early exit if only one group should be used
    if len(healthy_id_subsets) == 0:
        return affected_id_subsets
    elif len(affected_id_subsets) == 0:
        return healthy_id_subsets

    folds = [healthy_id_subsets.pop() + affected_id_subsets.pop() for i in range(num_folds)]

    while len(affected_id_subsets) > 0:
        # Flatten remaining
        affected_id_subsets = [aff_id for sublist in affected_id_subsets for aff_id in sublist]
        remaining_len = len(affected_id_subsets)
        for i in range(remaining_len):
            folds[i].append(affected_id_subsets.pop())

    for fold in folds:
        rand.shuffle(fold)

    return folds


def cross_val(data, labels, model, num_folds, healthy_ids, affected_ids, save_dir):
    '''
    Perform Cross-Validation for given dataset and subjects.
    :param data: NumPy array containing data of subjects.
    :param labels: NumPy array containing ground-truth labels.
    :param model: Classification model to train and validate (ex. kNN, SVM, DT, etc.)
    :param num_folds: Int, represents number of folds
    :param healthy_ids: List of String, ids of healthy subjects
    :param affected_ids: List of String, ids of affected subjects
    :param save_dir: String, path to directory where results should be saved
    '''

    main_save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Create results directory if not already made
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    folds = create_cv_folds(healthy_ids, affected_ids, num_folds)

    for idx in range(len(folds)):
        print(f'\n----- PERFORMING CROSS-VALIDATION FOR FOLD: {idx+1} -----\n')
        # Create training set
        train_ids = [sub_id for sublist in folds[:idx] + folds[idx+1:] for sub_id in sublist]

        train_data = [data[id] for id in train_ids]
        train_data = np.concatenate(train_data, 0)
        train_labels = [labels[id] for id in train_ids]
        train_labels = np.concatenate(train_labels, 0)

        # Create validation set
        val_data = np.concatenate([data[id] for id in folds[idx]])
        val_labels = np.concatenate([labels[id] for id in folds[idx]])

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
    data_path = cfg['DATA_PATH']
    healthy_subjects = cfg['HEALTHY_SUBJECTS']
    healthy_subjects = []
    affected_subjects = cfg['AFFECTED_SUBJECTS']
    subject_nums = healthy_subjects + affected_subjects
    save_dir = cfg['SAVE_MODEL_PATH']
    data_col = cfg['DATA_COL_NAME']
    label_col = cfg['LABEL_COL_NAME']
    labels_needed = cfg['GRASP_LABELS']
    emg_locs = cfg['EMG_ELECTRODE_LOCS']
    num_folds = cfg['CV_FOLDS']

    knn_clf = KNeighborsClassifier(n_jobs=-1)
    svm_clf = SVC(verbose=1)
    dt_clf = DecisionTreeClassifier()
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 10))
    # models = {'KNN': knn_clf,
    #           'SVM': svm_clf,
    #           'DecisionTree': dt_clf,
    #           'MLP': mlp_clf}
    models = {'MLP': mlp_clf}

    dataset_params = {'data_path': data_path,
                      'subject_nums': subject_nums,
                      'data_col': data_col,
                      'label_col': label_col,
                      'labels_needed': labels_needed,
                      'emg_locs': emg_locs}

    # all_subject_emg, all_subject_labels = get_all_subject_data(**dataset_params)
    with open(os.path.join(data_path, 'grasp_2_5_data.pkl'), 'rb') as handle:
        all_subject_emg = pickle.load(handle)
    with open(os.path.join(data_path, 'grasp_2_5_labels.pkl'), 'rb') as handle:
        all_subject_labels = pickle.load(handle)

    for model_name in models.keys():
        print(f'VALIDATING MODEL: {model_name}')
        model = models[model_name]
        save_dir_model = os.path.join(save_dir, model_name + '_' + str(datetime.datetime.now()))
        if not os.path.exists(save_dir_model):
            os.makedirs(save_dir_model)
        cross_val(all_subject_emg, all_subject_labels, model, num_folds, healthy_subjects, affected_subjects,
                  save_dir_model)

    print('Done.')
