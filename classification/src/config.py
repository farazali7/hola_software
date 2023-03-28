import os

cfg = {
    'DATASETS': {
        'NINAPRO_DB10': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db10',
            'FORMATTED_DATA_PATH': 'data/formatted/iter4/ninapro_db10',
            'PROCESSED_DATA_PATH': 'data/processed/iter6/ninapro_db10',
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
            'ELECTRODE_IDS': [3, 4, 5, 0, 1],  # Electrodes 3,4,5 are near bottom and 0,1 are near top, zero-indexed
            'GRASP_LABELS': [1, 2],  # 0 = hand relax, 1 = medium wrap (TVG), 2 = lateral grasp (pinch)
            'SAMPLING_FREQ': 1926
        },
        'GRABMYO': {
            'RAW_DATA_PATH': 'data/raw/grabmyo/open_hand',
            'FORMATTED_DATA_PATH': 'data/formatted/iter5/grabmyo_openhand',
            'PROCESSED_DATA_PATH': 'data/processed/iter7/grabmyo_openhand',
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
            #  In original indexing (GM to NP): bot: (1->4) (2->5) (3->6) top: (7->1) (8->2),
            'ELECTRODE_IDS': [[0, 8], [1, 9], [2, 10], [6, 14], [7, 15]],
            'SAMPLING_FREQ': 2048
        }
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
    'WINDOW_SIZE': 52,
    'WINDOW_OVERLAP_SIZE': 26,
    'COMBINE_CHANNELS': False,
    'STANDARDIZE': False,  # Set to False to normalize data into [-1, 1] range instead (MaxAbsScaler)
    'FEATURE_EXTRACTION_FUNC': 'feature_set_2',

    'MODEL_ARCHITECTURE': 'CNN',
    'EXPERIMENT_TYPE': 'train',

    # Training & validation args
    'BATCH_SIZE': 256,
    'EPOCHS': 40,
    'LR': 0.001,
    'SHUFFLE': True,
    'NUM_WORKERS': os.cpu_count(),

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

    # Data partitioning args
    'CV_FOLDS': 5,
    'TEST_SET_PERCENTAGE': 0.1,

    'CLASSES': ['OH', 'TVG', 'LP'],  # In label order

    'GLOBAL_SEED': 7,

    'HYPERPARAMETER_SEARCH': {
        'SWEEP_SETTINGS': {
            "method": "bayes",
            "metric": {
                "name": "Mean CV F1-Score",
                "goal": "maximize"
            },
        },
        'N_EVALS': 3,
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
        }
    },

    'WANDB': {
        'PROJECT': 'HOLA',
        'ENTITY': 'fali'
    }
}
