import datetime
import os
import pickle
import random

import numpy as np
import torch.utils.data
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from classification.utils.data_processing import get_all_subject_data, save_formatted_data
from classification.config import cfg
from classification.emg.models import MLP_MODEL, CNN_MODEL


def train_nn_model(train_data, train_labels, val_data, val_labels, model, batch_size, num_epochs, learning_rate, save_path):
    '''
    Train a neural network with an optimizer and specified data_processing/parameters.
    :param train_data: Array of input training data_processing
    :param train_labels: Array of ground-truth training labels
    :param val_data: Array of input validation data_processing
    :param val_labels: Array of ground-truth validation labels
    :param model: PyTorch Module neural network
    :param batch_size: Int, batch size of data_processing at each forward pass
    :param num_epochs: Int, number of epochs to train for
    :param learning_rate: Float, learning rate for model
    :param save_path: String, path to save best model over all epochs
    '''
    # Create Torch dataloader
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0
    model_save_path = os.path.join(save_path, 'model.pth')
    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs+1):
        train_loss = model.fit(train_loader, optimizer, epoch)
        train_losses.append(train_loss.detach().numpy())
        accuracy, val_loss = model.evaluate(val_loader)
        val_losses.append(val_loss.detach().numpy())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print('Saving...')
            torch.save(model.state_dict(), model_save_path)

    # Plot train and val loss
    epochs_x = range(1, num_epochs+1)
    plt.plot(epochs_x, train_losses, label='Train')
    plt.plot(epochs_x, val_losses, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    print(f'Best accuracy: {best_accuracy}')


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
    affected_in_fold = affected_in_fold + 1 if affected_in_fold == 0 else affected_in_fold
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


def cross_val(data, labels, model, num_folds, healthy_ids, affected_ids, batch_size, num_epochs, learning_rate,
              save_dir, train_ml=False):
    '''
    Perform Cross-Validation for given dataset and subjects.
    :param data: NumPy array containing data_processing of subjects.
    :param labels: NumPy array containing ground-truth labels.
    :param model: Classification model to train and validate (ex. kNN, SVM, DT, etc.)
    :param num_folds: Int, represents number of folds
    :param healthy_ids: List of String, ids of healthy subjects
    :param affected_ids: List of String, ids of affected subjects
    :param batch_size: Int, batch size of data_processing at each forward pass
    :param num_epochs: Int, number of epochs to train for
    :param learning_rate: Float, learning rate for model
    :param save_dir: String, path to directory where results should be saved
    :param train_ml: Boolean, whether to train a classical SkLearn model or an NN
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    folds = create_cv_folds(healthy_ids, affected_ids, num_folds)

    for idx in range(1, len(folds)):
        print(f'\n----- PERFORMING CROSS-VALIDATION FOR FOLD: {idx+1} -----\n')
        # Create training set
        train_ids = [sub_id for sublist in folds[:idx] + folds[idx+1:] for sub_id in sublist]

        train_data = [data[id] for id in train_ids]
        train_data = np.concatenate(train_data, 0)
        train_labels = [labels[id] for id in train_ids]
        train_labels = np.concatenate(train_labels, 0)[..., np.newaxis]

        train_data, train_labels = unison_shuffle(train_data, train_labels)

        # Create validation set
        val_data = np.concatenate([data[id] for id in folds[idx]])
        val_labels = np.concatenate([labels[id] for id in folds[idx]])[..., np.newaxis]

        val_data, val_labels = unison_shuffle(val_data, val_labels)

        # Fit to model
        if train_ml:
            model.fit(train_data, train_labels.ravel())
        else:
            train_nn_model(train_data, train_labels, val_data, val_labels, model, batch_size, num_epochs, learning_rate,
                           save_dir)

        # Predict with model
        preds = model.predict(val_data)

        # Save model, preds, and val labels
        np.savetxt(os.path.join(save_dir, f'cv{idx+1}_preds.csv'), preds, delimiter=',')
        np.savetxt(os.path.join(save_dir, f'cv{idx+1}_val_labels.csv'), val_labels, delimiter=',')
        pickle.dump(model, open(os.path.join(save_dir, f'cv{idx+1}_model.pkl'), 'wb'))

        print('Done.')


def map_label_values(labels, new_vals):
    '''
    Helper function to map array of labels to new values (ex. converting ordinal to binary)
    :param labels: Array of labels.
    :param new_vals: Dictionary containing mapping of all values in labels.
    :return: New list of labels with updated values.
    '''
    k = np.array(list(new_vals.keys()))
    v = np.array(list(new_vals.values()))
    mapping_ar = np.zeros(k.max()+1, dtype=v.dtype)
    mapping_ar[k] = v
    out = mapping_ar[labels]

    return out


def unison_shuffle(arr_a, arr_b):
    '''
    Shuffle two arrays in unison.
    :param arr_a: Array a
    :param arr_b: Array b
    :return: Shuffled arrays a and b
    '''
    assert len(arr_a) == len(arr_b)
    np.random.seed(4)
    p = np.random.permutation(len(arr_a))
    return arr_a[p], arr_b[p]


def create_dataset_from_dict(data_dict, labels_dict, features_list):
    '''
    Create final data_processing and labels datasets beased on requested features and initial dicts.
    :param data_dict: Dictionary containing all feature data_processing by subject
    :param labels_dict: Dictionary containing all labels by subject
    :param features_list: List of desired features to retain for dataset
    :return: Dictionaries of data_processing and labels containing pertinent feature data_processing
    '''
    final_data = {}
    final_labels = {}
    for subject in data_dict:
        final_data[subject] = np.concatenate([data_dict[subject][f] for f in features_list])
        final_labels[subject] = np.concatenate([labels_dict[subject][f] for f in features_list])

    return final_data, final_labels


if __name__ == "__main__":
    data_path = cfg['DATA_PATH']
    data_file = cfg['DATA_FILE']
    label_file = cfg['LABEL_FILE']
    healthy_subjects = cfg['HEALTHY_SUBJECTS']
    # healthy_subjects = []
    affected_subjects = cfg['AFFECTED_SUBJECTS']
    affected_subjects = []
    subject_nums = healthy_subjects + affected_subjects
    save_dir = cfg['SAVE_MODEL_PATH']
    data_col = cfg['DATA_COL_NAME']
    label_col = cfg['LABEL_COL_NAME']
    labels_needed = cfg['GRASP_LABELS']
    emg_locs = cfg['EMG_ELECTRODE_LOCS']
    num_folds = cfg['CV_FOLDS']
    window_size = cfg['WINDOW_SIZE']
    window_overlap_size = cfg['WINDOW_OVERLAP_SIZE']
    label_map = cfg['LABEL_MAP']
    features_list = cfg['FEATURES']

    batch_size = cfg['BATCH_SIZE']
    num_epochs = cfg['EPOCHS']
    learning_rate = cfg['LR']

    dataset_params = {'data_path': data_path,
                      'subject_nums': subject_nums,
                      'data_col': data_col,
                      'label_col': label_col,
                      'labels_needed': labels_needed,
                      'emg_locs': emg_locs,
                      'window_size': window_size,
                      'window_overlap_size': window_overlap_size}

    model = MLP_MODEL()

    train_ml = False
    # model = SVC(verbose=1)
    # model = DecisionTreeClassifier()
    # model = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=10)


    with open(os.path.join(data_path, data_file), 'rb') as handle:
        all_subject_emg = pickle.load(handle)
    with open(os.path.join(data_path, label_file), 'rb') as handle:
        all_subject_labels = pickle.load(handle)

    # Create final feature and label dataset
    data, labels = create_dataset_from_dict(all_subject_emg, all_subject_labels, features_list)

    # Convert labels
    for subject in labels.keys():
        labels[subject] = map_label_values(labels[subject], label_map)

    save_dir_model = os.path.join(save_dir, 'MLP_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    cross_val(data, labels, model, num_folds, healthy_subjects, affected_subjects,
              batch_size, num_epochs, learning_rate, save_dir_model, train_ml)

    print('Done.')
