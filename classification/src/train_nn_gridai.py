import datetime
import os
import argparse

import pandas as pd
import torch.utils.data

from classification.src.config import cfg
from classification.src.models.models import get_model, load_model_from_checkpoint
from classification.src.utils.experimentation import partition_dataset, create_equal_folds, \
    compute_class_weights, flatten_dict, aggregate_predictions
from classification.src.utils.data_pipeline import convert_to_full_paths, load_and_concat
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from classification.src.constants import CALLBACKS

from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import wandb

from copy import deepcopy
from functools import partial

GLOBAL_SEED = cfg['GLOBAL_SEED']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_callbacks(kwargs):
    callbacks = []
    try:
        for callback_name in kwargs:
            callback_cls = CALLBACKS[callback_name]
            callback = callback_cls(**kwargs[callback_name])
            callbacks.append(callback)
    except Exception as e:
        print(e)
        print('Could not define all callbacks.')

    return callbacks


def train_nn_model(train_data, train_labels, val_data, val_labels, model_def,
                   model_args, trainer_args, data_loader_args, callback_args, num_epochs,
                   gpus, logger):
    """
    Train a neural network with an optimizer and specified data_pipeline/parameters.
    :param train_data: List containing data IDs for training subjects
    :param val_data: List containing data IDs for validation subjects
    :param model_def: Specified model architecture
    :Param model_args: Dictionary of keyword arguments for model
    :param trainer_args: Dictionary of keyword arguments for trainer like learning rate, class weights, etc.
    :param data_loader_args: Dictionary of keyword arguments such as batch size, shuffling, num workers, etc.
    :param callback_args: Dictionary of callback specifications
    :param num_epochs: Int for number of max epochs
    :param gpus: Int for number of gpus
    :param logger: Logger object
    """
    # Create Torch dataloaders
    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels).to(torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_args)

    val_data_loader_args = data_loader_args.copy()
    val_data_loader_args['shuffle'] = False
    val_data = torch.Tensor(val_data)
    val_labels = torch.Tensor(val_labels).to(torch.long)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_data_loader_args)

    # Compute class weights
    class_weights = compute_class_weights(train_labels).to(device)
    trainer_args['class_weights'] = class_weights

    # Get model
    model = get_model(model_def, model_args, trainer_args)

    # Callbacks
    callbacks = define_callbacks(callback_args)

    # Trainer
    trainer = Trainer(callbacks=callbacks, max_epochs=num_epochs, deterministic=True, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load and return best model
    best_model_ckpt = trainer.checkpoint_callback.best_model_path
    trained_model = load_model_from_checkpoint(best_model_ckpt)

    val_out = trainer.predict(trained_model, dataloaders=val_loader)
    val_preds, val_targets = aggregate_predictions(val_out)
    metrics_def = trainer_args['metrics'].clone()
    val_metrics = metrics_def(val_preds, val_targets)

    # Flatten
    scalar_val_metrics = flatten_dict(val_metrics, trainer_args['classes'])

    return scalar_val_metrics, best_model_ckpt


def cross_val(data, model_def, training_params, num_folds, logger, save_dir=None):
    '''
    Perform Cross-Validation for given dataset and subjects.
    :param data: List of data sources to use
    :param model_def: String specifying which model architecture to use
    :param num_folds: Int, represents number of folds
    :param training_params: Dictionary of training hyperparameters
    :param logger: Logger object
    :param save_dir: String, path to directory where results should be saved
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    folds = create_equal_folds(data, num_folds, seed=GLOBAL_SEED, save_dir=save_dir)

    # Store fold num and checkpoint path to best model based on monitor metric
    all_fold_metrics = {}
    for idx in range(0, len(folds)):
        print(f'\n----- PERFORMING CROSS-VALIDATION FOR FOLD: {idx + 1} -----\n')
        # Get training folds
        train_ids = [sub_id for sublist in folds[:idx] + folds[idx + 1:] for sub_id in sublist]

        print('LOADING TRAINING AND VALIDATION DATA...')
        # Load all participant training data
        all_train_x, all_train_y = load_and_concat(train_ids, ext='.pkl')

        # Get validation fold
        val_ids = folds[idx]

        # Load all participant validation data
        all_val_x, all_val_y = load_and_concat(val_ids, ext='.pkl')

        # Set fold-specific training args
        fold_num = idx + 1
        monitor_metric_prefix = 'fold_' + str(fold_num) + '/'
        training_params_fold = deepcopy(training_params)
        training_params_fold['trainer_args']['fold'] = fold_num
        training_params_fold['callback_args']['MODEL_CHECKPOINT']['filename'] = \
            training_params_fold['callback_args']['MODEL_CHECKPOINT']['filename'] + '--fold=' + str(fold_num)
        training_params_fold['callback_args']['MODEL_CHECKPOINT']['monitor'] = monitor_metric_prefix + \
                                                                               training_params_fold['callback_args'][
                                                                                   'MODEL_CHECKPOINT']['monitor']
        if 'EARLY_STOPPING' in training_params_fold['callback_args']:
            training_params_fold['callback_args']['EARLY_STOPPING']['monitor'] = monitor_metric_prefix + \
                                                                                 training_params_fold['callback_args'][
                                                                                     'EARLY_STOPPING']['monitor']
        val_metrics, fold_ckpt = train_nn_model(train_data=all_train_x,
                                                train_labels=all_train_y,
                                                val_data=all_val_x,
                                                val_labels=all_val_y,
                                                model_def=model_def,
                                                **training_params_fold,
                                                logger=logger)

        all_fold_metrics[fold_num] = {}
        for metric_name in val_metrics:
            all_fold_metrics[fold_num][metric_name] = val_metrics[metric_name]
        all_fold_metrics[fold_num]['best_ckpt_path'] = fold_ckpt

    return all_fold_metrics


def train_single(train_set, model_def, training_params, num_folds, logger, save_dir=None):
    """
    Train a single model using the parameters specified.
    :param train_set: List of data sources (ex. subject IDs) for training
    :param model_def: String specifying which model architecture to use
    :param training_params: Dictionary of training hyperparameters
    :param num_folds: Int for number of k in cross-validation
    :param logger: Logger object
    :param save_dir: String to model saving directory
    """

    # Perform k-fold cross-validation on the training set to optimize hyperparameters
    all_cv_metrics = cross_val(train_set, model_def, training_params,
                               num_folds=num_folds, logger=logger,
                               save_dir=save_dir)

    # Create full and mean cross-validation dataframes for logging
    path_col = 'best_ckpt_path'
    raw_cv_df = pd.DataFrame(all_cv_metrics).T
    all_cv_df = pd.DataFrame(raw_cv_df.loc[:, raw_cv_df.columns != path_col]).astype('float')

    mean_cv_df = pd.DataFrame(all_cv_df.mean(axis=0)).T
    mean_cv_df.index.name = 'Value'
    mean_cv_df['Best Fold'] = all_cv_df['Macro F1-Score'].idxmax()
    mean_cv_df[path_col] = raw_cv_df[path_col][mean_cv_df['Best Fold']].values

    all_cv_df[path_col] = raw_cv_df[path_col]
    all_cv_df.insert(0, 'Fold', all_cv_df.index)

    logger.log_table('Full Cross-Validation Results', dataframe=all_cv_df)
    logger.log_table('Mean Cross-Validation Results', dataframe=mean_cv_df)
    logger.experiment.log({'Mean CV F1-Score': mean_cv_df['Macro F1-Score'].values[0]})

    # Return best model score and path
    best_score = all_cv_df['Macro F1-Score'].max()
    best_model_path = mean_cv_df[path_col].values[0]

    return best_score, best_model_path


def perform_sweep_iter(train_set, test_set, model_def, trainer_args, callback_args, data_loader_args, num_epochs,
                       num_folds, gpus, save_dir):
    """
    Perform an iteration of a hyperparameter sweep.
    :param train_set: List of training data from data sources
    :param test_set: List of test samples used for holdout
    :param model_def: String, main architecture for model to use
    :param trainer_args: Dictionary of trainer keyword arguments
    :param callback_args: Dictionary of callback keyword arguments
    :param data_loader_args: Dictionary of data loader keyword arguments
    :param num_epochs: Int for max number of epochs per train run
    :param num_folds: Int for number of k-fold cross validation folds
    :param gpus: Int for number of gpus to train on
    :param save_dir: String for path to save
    """
    with wandb.init(project=cfg['WANDB']['PROJECT'], entity=cfg['WANDB']['ENTITY']):
        config = wandb.config

        wandb_logger = WandbLogger(project=cfg['WANDB']['PROJECT'],
                                   entity=cfg['WANDB']['ENTITY'],
                                   save_dir=save_dir,
                                   log_model=True)

        model_args = dict(config)

        trainer_args['learning_rate'] = config.learning_rate

        training_params = {'model_args': model_args,
                           'trainer_args': trainer_args,
                           'data_loader_args': data_loader_args,
                           'callback_args': callback_args,
                           'num_epochs': num_epochs,
                           'gpus': gpus}

        model_score, model_path = train_single(train_set=train_set,
                                               model_def=model_def,
                                               training_params=training_params,
                                               num_folds=num_folds,
                                               logger=wandb_logger,
                                               save_dir=save_dir)

        # Get test set performance
        all_test_x, all_test_y = load_and_concat(test_set, ext='.pkl')

        test_data = torch.Tensor(all_test_x)
        test_labels = torch.Tensor(all_test_y).to(torch.long)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

        test_data_loader_args = deepcopy(data_loader_args)
        test_data_loader_args['shuffle'] = False
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_data_loader_args)

        trained_model = load_model_from_checkpoint(checkpoint_path=model_path)
        trainer = Trainer()

        test_out = {}
        test_out = trainer.predict(model=trained_model, dataloaders=test_loader)
        test_preds, test_targets = aggregate_predictions(test_out)
        metrics_c = trainer_args['metrics'].clone()
        test_metrics = metrics_c(test_preds, test_targets)

        scalar_test_metrics = flatten_dict(test_metrics, trainer_args['classes'])

        # Create df and log test_metrics as a table
        test_metrics_df = pd.DataFrame(scalar_test_metrics, index=[0])
        test_metrics_df['model_ckpt'] = model_path
        wandb_logger.log_table('Test Metrics', dataframe=test_metrics_df)


def perform_experiment(args):
    """
    Perform a model training experiment.
    :param args: Passed in from grid.ai script.
    """
    seed_everything(cfg['GLOBAL_SEED'])

    wandb_entity = cfg['WANDB']['ENTITY']
    wandb_project = cfg['WANDB']['PROJECT']

    # Ninapro DB10 data
    np_cfg = cfg['DATASETS']['NINAPRO_DB10']
    np_processed_data_path = os.path.join(args.data_dir, 'ninapro_db10')
    print(np_processed_data_path)
    np_healthy_subjects = np_cfg['HEALTHY_SUBJECTS']
    np_healthy_subjects = convert_to_full_paths(np_healthy_subjects, np_processed_data_path)
    np_affected_subjects = np_cfg['AFFECTED_SUBJECTS']
    np_affected_subjects = convert_to_full_paths(np_affected_subjects, np_processed_data_path)

    # GrabMyo data
    gm_cfg = cfg['DATASETS']['GRABMYO']
    gm_processed_data_path = os.path.join(args.data_dir, 'grabmyo_openhand')
    gm_healthy_subjects = gm_cfg['HEALTHY_SUBJECTS']
    gm_healthy_subjects = ['S' + str(x + 115) for x in gm_healthy_subjects]
    gm_healthy_subjects = convert_to_full_paths(gm_healthy_subjects, gm_processed_data_path)

    # data_sources = [np_healthy_subjects, np_affected_subjects, gm_healthy_subjects]
    data_sources = [np_healthy_subjects, gm_healthy_subjects]
    # data_sources = [gm_healthy_subjects]

    # DATALOADER ARGS #
    batch_size = cfg['BATCH_SIZE']
    shuffle = cfg['SHUFFLE']
    num_workers = cfg['NUM_WORKERS']

    data_loader_args = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}

    # MODEL AND SAVE PATH #
    # Saving paths (model, splits, checkpoints)
    model_def = cfg['MODEL_ARCHITECTURE']

    save_dir = os.path.join(cfg['SAVE_MODEL_PATH'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # NUM EPOCHS AND FOLDS #
    num_epochs = cfg['EPOCHS']
    num_folds = cfg['CV_FOLDS']

    # TRAINER ARGS #
    classes = cfg['CLASSES']
    trainer_args = {'classes': classes}

    # Metrics
    num_classes = len(classes)
    metrics = MetricCollection({
        'Accuracy': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
        'Multiclass Recall': Recall(task='multiclass', num_classes=num_classes, average=None),
        'Macro Recall': Recall(task='multiclass', num_classes=num_classes, average='macro'),
        'Multiclass Precision': Precision(task='multiclass', num_classes=num_classes, average=None),
        'Macro Precision': Precision(task='multiclass', num_classes=num_classes, average='macro'),
        'Multiclass F1-Score': F1Score(task='multiclass', num_classes=num_classes, average=None),
        'Macro F1-Score': F1Score(task='multiclass', num_classes=num_classes, average='macro'),
    })
    trainer_args['metrics'] = metrics

    # CALLBACK ARGS #
    callback_args = cfg['CALLBACKS']
    callback_args['MODEL_CHECKPOINT']['dirpath'] = save_dir

    test_set_percentage = cfg['TEST_SET_PERCENTAGE']
    train_set, test_set = partition_dataset(data_sources, test_percentage=test_set_percentage, seed=GLOBAL_SEED,
                                            save_dir=save_dir)

    sweep_cfg = cfg['HYPERPARAMETER_SEARCH']['SWEEP_SETTINGS']
    sweep_cfg['parameters'] = cfg['HYPERPARAMETER_SEARCH'][model_def]

    sweep_id = wandb.sweep(sweep_cfg, project=wandb_project, entity=wandb_entity)

    # Perform hyperparameter sweep and collect best scores
    wandb.agent(sweep_id, function=partial(perform_sweep_iter,
                                           train_set=train_set,
                                           test_set=test_set,
                                           model_def=model_def,
                                           trainer_args=trainer_args,
                                           callback_args=callback_args,
                                           data_loader_args=data_loader_args,
                                           num_epochs=num_epochs,
                                           num_folds=num_folds,
                                           gpus=args.gpus,
                                           save_dir=save_dir), count=cfg['HYPERPARAMETER_SEARCH']['N_EVALS'])

    # Stop sweep
    wandb.api.stop_sweep(sweep=sweep_id, entity=wandb_entity, project=wandb_project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0,
                        help='number of gpus to use for training')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='strategy to use for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='maximum number of epochs for training')
    parser.add_argument('--data_dir', type=str, default='/datastores/cifar5',
                        help='the directory to load data from')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='the learning rate to use during model training')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='the optimizer to use during model training')
    args = parser.parse_args()
    perform_experiment(args)
    print('Done.')
