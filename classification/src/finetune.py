import numpy as np
import torch
import os
import datetime
from copy import deepcopy
from pytorch_lightning.trainer import Trainer
from torchmetrics import MetricCollection, Precision, Recall, F1Score

from classification.src.config import cfg
from classification.src.utils.data_pipeline import convert_to_full_paths, load_and_concat
from classification.src.models.models import load_model_from_checkpoint, get_model
from classification.src.utils.experimentation import define_callbacks, aggregate_predictions, compute_class_weights

import pandas as pd


def adjust_subject_paths(subjects):
    subject_ids = [s.split('/')[-1] for s in subjects]

    np_healthy_subjects = cfg['DATASETS']['NINAPRO_DB10']['HEALTHY_SUBJECTS']
    np_process_path = cfg['DATASETS']['NINAPRO_DB10']['PROCESSED_DATA_PATH']

    gm_healthy_subjects = cfg['DATASETS']['GRABMYO']['HEALTHY_SUBJECTS']
    gm_process_path = cfg['DATASETS']['GRABMYO']['PROCESSED_DATA_PATH']

    test_set_subjects = []
    for id in subject_ids:
        base_path = np_process_path if id in np_healthy_subjects else gm_process_path
        path = convert_to_full_paths([id], base_path)
        test_set_subjects += path

    return test_set_subjects


def segregate_data_by_reps(subject):
    # Load their data
    test_x, test_y = load_and_concat([subject], ext='.pkl', include_uid=False, remove_trial_dim=False)

    # Perform one iteration of fine-tuning on X reps
    # Get training set (X reps for each grasp)
    unique_classes = np.unique(test_y)

    # For each class segregate by rep number
    segregated_data, segregated_labels = [], []
    for cls in unique_classes:
        cls_idxs = np.where(test_y == cls)[0]
        cls_samples = test_x[cls_idxs]

        cls_reps = np.unique(cls_samples[..., -1])

        cls_rep_data, cls_rep_labels = [], []
        for rep_id in cls_reps:
            # Flatten feature dims
            cls_sampled_reshaped = cls_samples.reshape((cls_samples.shape[0], -1, cls_samples.shape[-1]))
            rep_idxs = np.where(cls_sampled_reshaped[..., -1][:, 0] == rep_id)[0]

            rep_data = cls_samples[rep_idxs]
            rep_labels = test_y[cls_idxs][rep_idxs]
            cls_rep_data.append(rep_data)
            cls_rep_labels.append(rep_labels)

        segregated_data.append(cls_rep_data)
        segregated_labels.append(cls_rep_labels)

    return segregated_data, segregated_labels


def split_data_by_reps(data, labels, num_reps, hard_lim=7):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for idx in range(len(data)):
        cls_data = data[idx]
        cls_labels = labels[idx]

        cls_train_reps = cls_data[:num_reps]
        cls_train_labels = cls_labels[:num_reps]

        cls_test_reps = cls_data[num_reps:hard_lim]
        cls_test_labels = cls_labels[num_reps:hard_lim]

        train_data += cls_train_reps
        train_labels += cls_train_labels
        test_data += cls_test_reps
        test_labels += cls_test_labels

    train_data = np.vstack(train_data)[..., :-1]
    train_labels = np.vstack(train_labels)

    test_data = np.vstack(test_data)[..., :-1]
    test_labels = np.vstack(test_labels)

    return train_data, train_labels, test_data, test_labels


def finetune(subject, res_df, base_save_dir):
    seg_data, seg_labels = segregate_data_by_reps(subject)

    num_reps = finetune_params['REPS']
    train_data, train_labels, test_data, test_labels = split_data_by_reps(seg_data, seg_labels, num_reps)

    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels).to(torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_args)

    # Setup trainer args
    save_dir = os.path.join(base_save_dir, subject.split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    callback_args = finetune_params['CALLBACKS']
    callback_args['MODEL_CHECKPOINT']['dirpath'] = save_dir
    callbacks = define_callbacks(callback_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # class_weights = compute_class_weights(train_labels).to(device)
    class_weights = torch.Tensor([1.0, 1.0, 1.0]).to(device)
    trainer_args['class_weights'] = class_weights

    model_pth = torch.load(finetune_params['CHECKPOINT_PATH'])
    trainer_args['prev_optimizer_state'] = model_pth['optimizer_state_dict'][0]

    model = get_model(model_name=cfg['MODEL_ARCHITECTURE'], model_args=model_args, trainer_args=trainer_args,
                      use_legacy=False)
    model.load_state_dict(model_pth['model_state_dict'])

    # trained_model = load_model_from_checkpoint(checkpoint_path=finetune_params['CHECKPOINT_PATH'], strict=False)
    trainer = Trainer(callbacks=callbacks, max_epochs=finetune_params['EPOCHS'], deterministic=True, logger=False)

    # Fine-tune
    trainer.fit(model=model, train_dataloaders=train_loader)

    # Load best model
    best_model_ckpt = trainer.checkpoint_callback.best_model_path
    finetuned_model = load_model_from_checkpoint(best_model_ckpt)

    # Setup test dataloader
    test_data = torch.Tensor(test_data)
    test_labels = torch.Tensor(test_labels).to(torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    test_data_loader_args = deepcopy(data_loader_args)
    test_data_loader_args['shuffle'] = False
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_data_loader_args)

    # Predict with best model
    test_out = trainer.predict(finetuned_model, dataloaders=test_loader)
    test_preds, test_targets = aggregate_predictions(test_out)
    metrics_def = metrics.clone()
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
    df['Subject'] = subject

    torch.save(df, os.path.join(save_dir, 'test_metrics.pth'))

    # Save results
    print(f'Done subject: {subject}')

    return res_df

    # min_loss = np.float('inf')
    # for epoch in range(finetune_params['EPOCHS']):
    #     print(f'\n---- Epoch: {epoch} ----\n')
    #     loss = model.fit(train_loader, epoch)
    #     if loss < min_loss:
    #         torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))


if __name__ == '__main__':
    finetune_params = cfg['FINETUNE']

    batch_size = finetune_params['BATCH_SIZE']
    shuffle = cfg['SHUFFLE']
    num_workers = cfg['NUM_WORKERS']
    data_loader_args = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}

    classes = cfg['CLASSES']

    # Metrics
    num_classes = len(classes)
    metrics = MetricCollection({
        'Multiclass Recall': Recall(task='multiclass', num_classes=num_classes, average=None),
        'Multiclass Precision': Precision(task='multiclass', num_classes=num_classes, average=None),
        'Multiclass F1-Score': F1Score(task='multiclass', num_classes=num_classes, average=None),
    })

    trainer_args = {'classes': classes,
                    'metrics': metrics,
                    'learning_rate': 0}

    model_args = {'dropout': 0}

    # Load test set subjects nums
    test_set_subjects_path = 'results/gridai/test_set.pkl'
    test_set_subjects = torch.load(test_set_subjects_path)

    if finetune_params['RUN_LOCALLY']:  # Adjust paths
        test_set_subjects = adjust_subject_paths(test_set_subjects)

    res_df = pd.DataFrame(columns=['Subject',
                                   'F1 OH', 'F1 TVG', 'F1 LP',
                                   'Precision OH', 'Precision TVG', 'Precision LP',
                                   'Recall OH', 'Recall TVG', 'Recall LP'])

    base_save_dir = os.path.join(cfg['SAVE_MODEL_PATH'], "finetune-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    for i, subject in enumerate(test_set_subjects[8:]):
        print(i)
        res_df = finetune(subject, res_df, base_save_dir=base_save_dir)

    res_df.to_csv('full_test_metrics.csv')

    print('Done fine-tuning.')
