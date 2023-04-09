import torch
import os
import datetime
from copy import deepcopy
from pytorch_lightning.trainer import Trainer
from torchmetrics import MetricCollection, Precision, Recall, F1Score

from classification.src.config import cfg
from classification.src.models.models import load_model_from_checkpoint, get_model
from classification.src.utils.experimentation import define_callbacks, aggregate_predictions, compute_class_weights, \
    majority_vote_transform, adjust_subject_paths, segregate_data_by_reps, split_data_by_reps

import pandas as pd


def finetune(subject, res_df, base_save_dir, reduce_lr=False, evaluate_by_mv=False, voters=None):
    seg_data, seg_labels = segregate_data_by_reps(subject)

    num_reps = finetune_params['REPS']
    train_data, train_labels, test_data, test_labels = split_data_by_reps(seg_data, seg_labels, num_reps)

    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels).to(torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_args, drop_last=True)

    # Setup trainer args
    save_dir = os.path.join(base_save_dir, subject[0].split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    callback_args = finetune_params['CALLBACKS']
    callback_args['MODEL_CHECKPOINT']['dirpath'] = save_dir
    callbacks = define_callbacks(callback_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # class_weights = compute_class_weights(train_labels).to(device)
    class_weights = torch.Tensor([0.6, 0.9, 0.9]).to(device)
    trainer_args['class_weights'] = class_weights

    model_pth = torch.load(finetune_params['CHECKPOINT_PATH'], map_location='cpu')
    trainer_args['prev_optimizer_state'] = deepcopy(model_pth['optimizer_states'][0])

    if reduce_lr:  # Reduce pretraining LR by factor of 10 for fine-tuning
        trainer_args['prev_optimizer_state']['param_groups'][0]['lr'] /= 10

    model = get_model(model_name=cfg['MODEL_ARCHITECTURE'], model_args=model_args, trainer_args=trainer_args,
                      use_legacy=False)
    model.load_state_dict(model_pth['state_dict'])

    # trained_model = load_model_from_checkpoint(checkpoint_path=finetune_params['CHECKPOINT_PATH'],
    #                                            metrics=trainer_args['metrics'].clone(), class_weights=class_weights)
    trainer = Trainer(callbacks=callbacks, max_epochs=finetune_params['EPOCHS'], deterministic=True, logger=False)

    # Fine-tune
    trainer.fit(model=model, train_dataloaders=train_loader)

    # Load best model
    best_model_ckpt = trainer.checkpoint_callback.best_model_path
    finetuned_model = load_model_from_checkpoint(best_model_ckpt, metrics=trainer_args['metrics'].clone())

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
    test_set_subjects_path = finetune_params['TEST_SET_SUBJECTS_PATH']
    test_set_subjects = torch.load(test_set_subjects_path)

    if finetune_params['ON_AMPUTEES']:
        amputee_ids = ['S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S107']
        for i in range(len(test_set_subjects)):
            if 'ninapro_db10' in test_set_subjects[i] and len(amputee_ids) > 0:
                curr_subject = test_set_subjects[i].split('/')[-1]
                amputee = test_set_subjects[i].split(curr_subject)[0] + amputee_ids.pop()
                test_set_subjects[i] = amputee

    if finetune_params['RUN_LOCALLY']:  # Adjust paths
        test_set_subjects = adjust_subject_paths(test_set_subjects)

    res_df = pd.DataFrame(columns=['Subject',
                                   'F1 OH', 'F1 TVG', 'F1 LP',
                                   'Precision OH', 'Precision TVG', 'Precision LP',
                                   'Recall OH', 'Recall TVG', 'Recall LP'])

    base_save_dir = os.path.join(cfg['SAVE_MODEL_PATH'], "finetune-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    pairs = [z for z in zip(test_set_subjects[::2], test_set_subjects[1::2])]
    for i, subject in enumerate(pairs):
        print(i)
        res_df = finetune(subject, res_df, base_save_dir=base_save_dir, reduce_lr=finetune_params['REDUCE_LR'],
                          evaluate_by_mv=finetune_params['PERFORM_MAJORITY_VOTING'], voters=finetune_params['VOTERS'])

    res_df.to_csv(os.path.join(base_save_dir, 'full_test_metrics.csv'))

    print('Done fine-tuning.')
