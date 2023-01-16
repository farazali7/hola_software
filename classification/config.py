
cfg = {
    'DATA_PATH': 'data/ninapro_db10',
    'SAVE_MODEL_PATH': 'results/models',
    'DATA_COL_NAME': 'emg',
    'LABEL_COL_NAME': 'regrasp',
    'HEALTHY_SUBJECTS': ['S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S020', 'S021',
                         'S022', 'S023', 'S024', 'S026', 'S027', 'S028', 'S029', 'S030', 'S031', 'S032', 'S033', 'S034',
                         'S035', 'S036', 'S037', 'S038', 'S039', 'S040'],
    'AFFECTED_SUBJECTS': ['S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S107', 'S108', 'S109', 'S110', 'S111',
                          'S112', 'S113', 'S114', 'S115'],
    'EMG_ELECTRODE_LOCS': [0, 1, 3, 5, 7],
    'GRASP_LABELS': [2, 5],
    'CV_FOLDS': 5,
    'AFFECT_HEALTHY_RATIO': 0.34
}
