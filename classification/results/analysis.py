import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pandas import read_csv
import os
from classification.utils.performance_metrics.metrics import get_accuracy, get_sensitivity, get_specificity, get_f1score


def get_cv_metrics(results_path):
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []

    num_vals = np.int32(len([n for n in os.listdir(results_path)]) / 3)

    for cv_num in range(1, num_vals+1):
        preds = read_csv(os.path.join(results_path, f'cv{cv_num}_preds.csv'), header=None, dtype=int)
        labels = read_csv(os.path.join(results_path, f'cv{cv_num}_val_labels.csv'), header=None, dtype=int)
        accuracies.append(get_accuracy(labels, preds))
        sensitivities.append(get_sensitivity(labels, preds))
        specificities.append(get_specificity(labels, preds))
        f1_scores.append(get_f1score(labels, preds))

    print(f'MEAN CV ACCURACY: {np.mean(accuracies)}')
    print(f'MEAN CV SENSITIVITY: {np.mean(sensitivities)}')
    print(f'MEAN CV RECALL: {np.mean(specificities)}')
    print(f'MEAN CV F1-SCORE: {np.mean(f1_scores)}')
    print('\n-------------------------------------------------------\n')


if __name__ == "__main__":
    results_path = 'models/MLP_20230117-205552'
    get_cv_metrics(results_path)

    print('Done')