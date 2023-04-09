import scipy
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from torchmetrics import MetricCollection, Precision, Recall, F1Score
import torch
import pandas as pd
import os
import datetime
import numpy as np

from classification.src.config import cfg
from classification.src.utils.experimentation import adjust_subject_paths, split_data_by_reps, segregate_data_by_reps, \
    aggregate_predictions, majority_vote_transform


def cmc(subject, res_df, model, base_save_dir, evaluate_by_mv=False, voters=None, metrics=None):
    save_dir = os.path.join(base_save_dir, subject[0].split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seg_data, seg_labels = segregate_data_by_reps(subject)

    num_reps = cm_comparison_params['REPS']
    train_data, train_labels, test_data, test_labels = split_data_by_reps(seg_data, seg_labels, num_reps)

    # Fix data and labels to 2D for classifiers
    train_labels = np.tile(train_labels, train_data.shape[1])
    train_data = train_data.reshape(-1, 25)
    train_labels = train_labels.reshape(-1)

    test_labels = np.tile(test_labels, test_data.shape[1])
    test_data = test_data.reshape(-1, 25)
    test_labels = test_labels.reshape(-1)

    # Shuffle the training data
    np.random.seed(cfg['GLOBAL_SEED'])
    p = np.random.permutation(len(train_data))
    train_data = train_data[p]
    train_labels = train_labels[p]

    # Fit to model
    model.fit(train_data, train_labels)

    # Predict with model
    preds = model.predict(test_data)

    test_preds, test_targets = torch.tensor(preds), torch.tensor(test_labels)
    metrics_def = metrics.clone()
    if evaluate_by_mv:
        assert voters is not None, "Param: 'voters' must be set to an integer if evaluating by majority vote."
        mv_preds, mv_targets = majority_vote_transform(test_preds, test_targets, voters=voters, drop_last=True)
        test_metrics = metrics_def(mv_preds, mv_targets)
    else:
        test_metrics = metrics_def(test_preds, test_targets)

    f1_scores = test_metrics['Multiclass F1-Score'].detach().numpy()
    precision_scores = test_metrics['Multiclass Precision'].detach().numpy()
    recall_scores = test_metrics['Multiclass Recall'].detach().numpy()

    row = [subject,
           f1_scores[0], f1_scores[1], f1_scores[2],
           precision_scores[0], precision_scores[1], precision_scores[2],
           recall_scores[0], recall_scores[1], recall_scores[2]]

    res_df.loc[len(res_df)] = row

    df = pd.DataFrame(test_metrics)
    df['Subject'] = subject[0] + '&' + subject[1]

    torch.save(df, os.path.join(save_dir, 'test_metrics.pth'))

    # Save results
    print(f'Done subject: {subject}')

    return res_df


if __name__ == "__main__":
    cm_comparison_params = cfg['CLASSICAL_MODEL_COMPARISON']
    classes = cfg['CLASSES']

    # Metrics
    num_classes = len(classes)
    metrics = MetricCollection({
        'Multiclass Recall': Recall(task='multiclass', num_classes=num_classes, average=None),
        'Multiclass Precision': Precision(task='multiclass', num_classes=num_classes, average=None),
        'Multiclass F1-Score': F1Score(task='multiclass', num_classes=num_classes, average=None),
    })

    # Load test set subjects nums
    test_set_subjects_path = cm_comparison_params['TEST_SET_SUBJECTS_PATH']
    test_set_subjects = torch.load(test_set_subjects_path)

    subject_type = 'healthy'
    if cm_comparison_params['ON_AMPUTEES']:
        amputee_ids = ['S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S107']
        subject_type = 'amputees'
        for i in range(len(test_set_subjects)):
            if 'ninapro_db10' in test_set_subjects[i] and len(amputee_ids) > 0:
                curr_subject = test_set_subjects[i].split('/')[-1]
                amputee = test_set_subjects[i].split(curr_subject)[0] + amputee_ids.pop()
                test_set_subjects[i] = amputee

    if cm_comparison_params['RUN_LOCALLY']:  # Adjust paths
        test_set_subjects = adjust_subject_paths(test_set_subjects)

    res_df = pd.DataFrame(columns=['Subject',
                                   'F1 OH', 'F1 TVG', 'F1 LP',
                                   'Precision OH', 'Precision TVG', 'Precision LP',
                                   'Recall OH', 'Recall TVG', 'Recall LP'])

    pairs = [z for z in zip(test_set_subjects[::2], test_set_subjects[1::2])]
    knn_clf = KNeighborsClassifier(n_jobs=-1)
    svm_clf = SVC(verbose=1)
    dt_clf = DecisionTreeClassifier()
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 10))
    models = {'KNN': knn_clf,
              'SVM': svm_clf,
              'DecisionTree': dt_clf,
              'MLP': mlp_clf}

    for model_name in models.keys():
        print(f'TRAINING & TESTING MODEL: {model_name}')
        model = models[model_name]
        prefix = f'{model_name}_{subject_type}'
        base_save_dir = os.path.join(cfg['SAVE_MODEL_PATH'], f"cmc_{prefix}" +
                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        res_df = pd.DataFrame(columns=['Subject',
                                       'F1 OH', 'F1 TVG', 'F1 LP',
                                       'Precision OH', 'Precision TVG', 'Precision LP',
                                       'Recall OH', 'Recall TVG', 'Recall LP'])
        for i, subject in enumerate(pairs):
            print(f'Subject: {i}')
            res_df = cmc(subject, res_df,
                         model=model,
                         base_save_dir=base_save_dir,
                         evaluate_by_mv=cm_comparison_params['PERFORM_MAJORITY_VOTING'],
                         voters=cm_comparison_params['VOTERS'],
                         metrics=metrics)

        res_df.to_csv(os.path.join(base_save_dir, f"{prefix}_"+'full_test_metrics.csv'))

    print('Done classical model comparison.')