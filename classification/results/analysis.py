import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pandas import read_csv
import os


def get_accuracy(labels, preds):
    return accuracy_score(labels, preds)


def get_precision(labels, preds):
    return precision_score(labels, preds, pos_label=5)


def get_sensitivity(labels, preds):
    return recall_score(labels, preds, pos_label=17)


def get_recall(labels, preds):
    return recall_score(labels, preds, pos_label=5)


def get_f1score(labels, preds):
    return f1_score(labels, preds, pos_label=5)


def get_cv_metrics(results_path):
    accuracies = []
    sensitivities = []
    recalls = []
    f1_scores = []

    for cv_num in range(1, 11):
        preds = read_csv(os.path.join(results_path, f'cv{cv_num}_preds.csv'), header=None, dtype=int)
        labels = read_csv(os.path.join(results_path, f'cv{cv_num}_val_labels.csv'), header=None, dtype=int)
        accuracies.append(get_accuracy(labels, preds))
        sensitivities.append(get_sensitivity(labels, preds))
        recalls.append(get_recall(labels, preds))
        f1_scores.append(get_f1score(labels, preds))

    print(f'MEAN CV ACCURACY: {np.mean(accuracies)}')
    print(f'MEAN CV SENSITIVITY: {np.mean(sensitivities)}')
    print(f'MEAN CV RECALL: {np.mean(recalls)}')
    print(f'MEAN CV F1-SCORE: {np.mean(f1_scores)}')
    print('\n-------------------------------------------------------\n')


if __name__ == "__main__":
    results_dir = "results/models"
    # # All features EMG
    # model_dirs = ['KNN/20221128-203653', 'SVM/20221128-203657', "DecisionTree/20221128-203717",
    #               'MLP/20221128-203731']

    # # All features IMU
    # model_dirs = ['KNN_IMU/20221129-161436', 'SVM_IMU/20221129-161443', "DecisionTree_IMU/20221129-161709",
    #               'MLP_IMU/20221129-161748']

    # # 3 TD features EMG
    # model_dirs = ['KNN/20221128-233705', 'SVM/20221128-233708', "DecisionTree/20221128-233717",
    #               'MLP/20221128-233724']

    # # 3 TD features IMU
    # model_dirs = ['KNN_IMU/20221129-174713', 'SVM_IMU/20221129-174717', "DecisionTree_IMU/20221129-174804",
    #               'MLP_IMU/20221129-174820']

    # # All features Both EMG and IMU
    # model_dirs = ['KNN_IMUEMG/20221129-191517', 'SVM_IMUEMG/20221129-191531', "DecisionTree_IMUEMG/20221129-192135",
    #               'MLP_IMUEMG/20221129-192220']

    # 3 TD features Both EMG and IMU
    model_dirs = ['KNN_IMUEMG/20221203-173623', 'SVM_IMUEMG/20221203-173628', "DecisionTree_IMUEMG/20221203-173751",
                  'MLP_IMUEMG/20221203-173817']


    for model_dir in model_dirs:
        model_name = model_dir.split('/')[0]
        print(f'\n----- METRICS FOR MODEL: {model_name} -----\n')
        model_res_path = os.path.join(results_dir, model_dir)
        get_cv_metrics(model_res_path)

    print('Done')