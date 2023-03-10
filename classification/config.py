
cfg = {
    'DATASETS': {
        'NINAPRO_DB10': {
            'RAW_DATA_PATH': 'data/raw/ninapro_db10',
            'FORMATTED_DATA_PATH': 'data/formatted/ninapro_db10',
            'PROCESSED_DATA_PATH': 'data/processed/ninapro_db10',
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
            'ELECTRODE_IDS': [3, 5, 0, 1, 7],  # Electrodes 0,1,7 are near top and 3,5 are near bottom, zero-indexed
            'GRASP_LABELS': [0, 1, 2],  # 0 = hand relax, 1 = medium wrap (TVG), 2 = lateral grasp (pinch)
            'SAMPLING_FREQ': 1926
        },
        'GRABMYO': {
            'RAW_DATA_PATH': 'data/raw/grabmyo/open_hand',
            'FORMATTED_DATA_PATH': 'data/formatted/grabmyo_openhand',
            'PROCESSED_DATA_PATH': 'data/processed/grabmyo_openhand',
            'HEALTHY_SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                 40, 41, 42, 43],
            #  First two pairs are near bottom and last three pairs are near top (all monopolar)
            #  Each in order of [closer to elbow, distal], zero-indexed
            #  In original indexing (GM to NP): top: (6->8) (7->1) (8->2), bot: (2->4) (4->6)
            'ELECTRODE_IDS': [[1, 9], [3, 11], [6, 14], [7, 15], [5, 13]],
            'SAMPLING_FREQ': 2048
        }
    },
    'DATA_FILE': 'grasp_2_5_w400_sepch_data.pkl',
    'LABEL_FILE': 'grasp_2_5_w400_sepch_labels.pkl',
    'SAVE_MODEL_PATH': 'results/models',

    # Preprocessing args
    'BUTTERWORTH_ORDER': 4,
    'BUTTERWORTH_FREQ': 20,
    'NOTCH_FREQ': 60,
    'QUALITY_FACTOR': 30.0,
    'TARGET_FREQ': 250,  # Frequency expected in real-time acquisition

    # Feature extraction args
    'WINDOW_SIZE': 60,
    'WINDOW_OVERLAP_SIZE': 30,
    'COMBINE_CHANNELS': False,
    'STANDARDIZE': False,  # Set to False to normalize data into [-1, 1] range instead (MaxAbsScaler)

    # Training & validation args
    'CV_FOLDS': 5,
    'BATCH_SIZE': 200,
    'EPOCHS': 10,
    'LR': 0.001,

    'LABEL_MAP': {2: 0,
                  5: 1},  # 2 is LP (negative class) and 5 is TVG (positive class)

    'FEATURES': ['rms', 'mav', 'var', 'mdwt']
}
