import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import read_csv
import os


def get_accuracy(labels, preds):
    return accuracy_score(labels, preds)


def get_precision(labels, preds):
    return precision_score(labels, preds, pos_label=5)


def get_recall(labels, preds):
    return recall_score(labels, preds, pos_label=5)


def get_f1score(labels, preds):
    return f1_score(labels, preds, pos_label=5)


def get_cv_metrics(results_path):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for cv_num in range(1, 11):
        preds = read_csv(os.path.join(results_path, f'cv{cv_num}_preds.csv'), header=None, dtype=int)
        labels = read_csv(os.path.join(results_path, f'cv{cv_num}_val_labels.csv'), header=None, dtype=int)
        accuracies.append(get_accuracy(labels, preds))
        precisions.append(get_precision(labels, preds))
        recalls.append(get_recall(labels, preds))
        f1_scores.append(get_f1score(labels, preds))

    print(f'MEAN CV ACCURACY: {np.mean(accuracies)}')
    print(f'MEAN CV PRECISION: {np.mean(precisions)}')
    print(f'MEAN CV RECALL: {np.mean(recalls)}')
    print(f'MEAN CV F1-SCORE: {np.mean(f1_scores)}')
    print('\n-------------------------------------------------------\n')


if __name__ == "__main__":
    results_dir = "results/"
    # All features
    model_dirs = ['KNN/20221128-203653', 'SVM/20221128-203657', "DecisionTree/20221128-203717",
                  'MLP/20221128-203731']

    # 3 TD features
    # model_dirs = ['KNN/20221128-233705', 'SVM/20221128-233708', "DecisionTree/20221128-233717",
    #               'MLP/20221128-233724']
    for model_dir in model_dirs:
        model_name = model_dir.split('/')[0]
        print(f'\n----- METRICS FOR MODEL: {model_name} -----\n')
        model_res_path = os.path.join(results_dir, model_dir)
        get_cv_metrics(model_res_path)

    print('Done')