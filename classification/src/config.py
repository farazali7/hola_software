import os

cfg = {
    'DATASETS': {
        'NINAPRO_DB10': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db10',
            'FORMATTED_DATA_PATH': 'data/formatted/iter8/ninapro_db10',
            'PROCESSED_DATA_PATH': 'data/processed/iter13/ninapro_db10',
            'DATA_COL_NAME': 'emg',
            'LABEL_COL_NAME': 'regrasp',
            'HEALTHY_SUBJECTS': ['S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S020',
                                 'S021',
                                 'S022', 'S023', 'S026', 'S027', 'S028', 'S029', 'S030', 'S031', 'S032', 'S033',
                                 'S034',
                                 'S035', 'S036', 'S037', 'S038', 'S039', 'S040'],
            'AFFECTED_SUBJECTS': ['S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S107', 'S108', 'S109', 'S110',
                                  'S111',
                                  'S112', 'S113', 'S114', 'S115'],
            'ELECTRODE_IDS': [3, 5, 0, 1, 6],  # Electrodes 0,1,6 are near top and 3,5 are near bottom, zero-indexed
            'GRASP_LABELS': [1, 2],  # 0 = hand relax, 1 = medium wrap (TVG), 2 = lateral grasp (pinch)
            'SAMPLING_FREQ': 1926
        },
        'GRABMYO': {
            'RAW_DATA_PATH': 'data/raw/grabmyo/open_hand',
            'FORMATTED_DATA_PATH': 'data/formatted/iter8/grabmyo_openhand',
            'PROCESSED_DATA_PATH': 'data/processed/iter13/grabmyo_openhand',
            'HEALTHY_SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                 40, 41, 42, 43],
            'GRASP_IDS': {
                'OH': 0,
                'TVG': 1,
                'LP': 2
            },  # Since 1-2 are taken by TVG & LP
            #  First two pairs are near bottom and last three pairs are near top (all monopolar)
            #  Each in order of [closer to elbow (proximal), distal], zero-indexed
            #  In original indexing (GM to NP): top: (6->7) (7->1) (8->2), bot: (2->4) (4->6)
            'ELECTRODE_IDS': [[1, 9], [3, 11], [6, 14], [7, 15], [5, 13]],
            'SAMPLING_FREQ': 2048
        },
        'NINAPRO_DB2': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db2',
            'FORMATTED_DATA_PATH': 'data/formatted/iter10/ninapro_db2',
            'PROCESSED_DATA_PATH': 'data/processed/iter12/ninapro_db2',
            'HEALTHY_SUBJECTS': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 27, 28,
                                 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            'ELECTRODE_IDS': [4, 5, 1, 2, 3],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            # 'ELECTRODE_IDS': [2, 4, 0, 6, 7],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            'GRASP_LABELS': [5, 5, 17],  # 5 = hand open (exB), 5 = medium wrap (TVG)(exC), 17 = lateral grasp (pinch)
            'SAMPLING_FREQ': 2000,
            'REGRASP_IDS': {
                'OH': 0,
                'TVG': 1,
                'LP': 2
            }
        },
        'NINAPRO_DB5': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db5',
            'FORMATTED_DATA_PATH': 'data/formatted/iter10/ninapro_db5',
            'PROCESSED_DATA_PATH': 'data/processed/iter12/ninapro_db5',
            'HEALTHY_SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'ELECTRODE_IDS': [4, 5, 1, 2, 3],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            # 'ELECTRODE_IDS': [2, 4, 0, 6, 7],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            'GRASP_LABELS': [5, 5, 17],  # 5 = hand open (exB), 5 = medium wrap (TVG)(exC), 17 = lateral grasp (pinch)
            'SAMPLING_FREQ': 200,
            'REGRASP_IDS': {
                'OH': 0,
                'TVG': 1,
                'LP': 2
            }
        },
        'NINAPRO_DB7': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db7',
            'FORMATTED_DATA_PATH': 'data/formatted/iter10/ninapro_db7',
            'PROCESSED_DATA_PATH': 'data/processed/iter12/ninapro_db7',
            'HEALTHY_SUBJECTS': [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
            'ELECTRODE_IDS': [4, 5, 1, 2, 3],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            # 'ELECTRODE_IDS': [2, 4, 0, 6, 7],  # Electrodes 2, 4 are near bot and 0, 6, 7 are near top, zero-indexed
            'GRASP_LABELS': [5, 5, 17],  # 5 = hand open (exB), 5 = medium wrap (TVG)(exC), 17 = lateral grasp (pinch)
            'SAMPLING_FREQ': 2000,
            'REGRASP_IDS': {
                'OH': 0,
                'TVG': 1,
                'LP': 2
            }
        },
    },
    'SAVE_MODEL_PATH': 'results/models',
    'SAVE_SPLITS_PATH': 'results/models',

    # Preprocessing args
    'BUTTERWORTH_ORDER': 2,
    'BUTTERWORTH_FREQ': [10, 100],
    'NOTCH_FREQ': 50,
    'QUALITY_FACTOR': 30.0,
    'TARGET_FREQ': 200,  # Frequency expected in real-time acquisition

    # Feature extraction args
    'WINDOW_SIZE': 38,
    'WINDOW_OVERLAP_SIZE': 29,
    'COMBINE_CHANNELS': False,
    'STANDARDIZE': False,  # Set to False to normalize data into [-1, 1] range instead (MaxAbsScaler)
    'FEATURE_EXTRACTION_FUNC': 'feature_set_6',

    'MODEL_ARCHITECTURE': 'CNN_ITER4',
    'EXPERIMENT_TYPE': 'train',

    # Training & validation args
    'BATCH_SIZE': 256,
    'EPOCHS': 40,
    'LR': 0.001,
    'SHUFFLE': True,
    'NUM_WORKERS': os.cpu_count(),

    # Data partitioning args
    'CV_FOLDS': 5,
    'TEST_SET_PERCENTAGE': 0.2,

    'CLASSES': ['OH', 'TVG', 'LP'],  # In label order

    'BATCH_SPECIFIC_TRAIN': False,  # If train batches should be specific to 1 subject at a time (ex. for AdaBN scheme)

    'GLOBAL_SEED': 7,

    'CALLBACKS': {
        # 'EARLY_STOPPING': {
        #     'monitor': 'val_Macro F1-Score',
        #     'min_delta': 0.001,
        #     'patience': 8
        # },
        'MODEL_CHECKPOINT': {
            'filename': '{epoch}--{val_Macro F1-Score:.2f}',
            'monitor': 'val_Macro F1-Score',
            'mode': 'max',
            'auto_insert_metric_name': True
        }
    },

    'HYPERPARAMETER_SEARCH': {
        'SWEEP_SETTINGS': {
            "method": "bayes",
            "metric": {
                "name": "Mean CV F1-Score",
                "goal": "maximize"
            },
        },
        'N_EVALS': 11,
        'MLP': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'MLP_ITER2': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER2': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER3': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER4': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        }
    },

    'WANDB': {
        'PROJECT': 'HOLA',
        'ENTITY': 'fali'
    },

    # Fine-tuning parameters such as whether to run script locally (adjust paths), how many reps to use for training
    'FINETUNE': {
        'RUN_LOCALLY': False,
        'ON_AMPUTEES': True,
        'REPS': 4,
        'TEST_SET_SUBJECTS_PATH': 'results/models/20230401-200055/test_set.pkl',
        'CHECKPOINT_PATH': 'results/models/20230401-200055/epoch=21--val_Macro F1-Score=0.00--fold=1-v1.ckpt',  # Pre-trained model
        'EPOCHS': 20,
        'BATCH_SIZE': 32,
        'CALLBACKS': {
            # 'EARLY_STOPPING': {
            #     'monitor': 'val_Macro F1-Score',
            #     'min_delta': 0.001,
            #     'patience': 8
            # },
            'MODEL_CHECKPOINT': {
                'filename': '{epoch}--{train_loss:.2f}',
                'monitor': 'train_loss',
                'mode': 'min',
                'auto_insert_metric_name': True
            }
        },
        'REDUCE_LR': True,
        'PERFORM_MAJORITY_VOTING': True,
        'VOTERS': 3
    },
}
