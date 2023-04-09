from classification.src.config import cfg
from classification.src.utils.data_pipeline import convert_to_full_paths


def adjust_subject_paths(subjects):
    subject_ids = [s.split('/')[-1] for s in subjects]

    np_healthy_subjects = cfg['DATASETS']['NINAPRO_DB10']['HEALTHY_SUBJECTS']
    np_affected_subjects = cfg['DATASETS']['NINAPRO_DB10']['AFFECTED_SUBJECTS']
    np_process_path = cfg['DATASETS']['NINAPRO_DB10']['PROCESSED_DATA_PATH']

    gm_healthy_subjects = ['S' + str(x + 115) for x in cfg['DATASETS']['GRABMYO']['HEALTHY_SUBJECTS']]
    gm_process_path = cfg['DATASETS']['GRABMYO']['PROCESSED_DATA_PATH']

    np2_healthy_subjects = ['np2_'+str(s) for s in cfg['DATASETS']['NINAPRO_DB2']['HEALTHY_SUBJECTS']]
    np2_process_path = cfg['DATASETS']['NINAPRO_DB2']['PROCESSED_DATA_PATH']

    np5_healthy_subjects = ['np5_'+str(s) for s in cfg['DATASETS']['NINAPRO_DB5']['HEALTHY_SUBJECTS']]
    np5_process_path = cfg['DATASETS']['NINAPRO_DB5']['PROCESSED_DATA_PATH']

    np7_healthy_subjects = ['np7_'+str(s) for s in cfg['DATASETS']['NINAPRO_DB7']['HEALTHY_SUBJECTS']]
    np7_process_path = cfg['DATASETS']['NINAPRO_DB7']['PROCESSED_DATA_PATH']

    test_set_subjects = []
    for id in subject_ids:
        if id in np_healthy_subjects or id in np_affected_subjects:
            base_path = np_process_path
        elif id in gm_healthy_subjects:
            base_path = gm_process_path
        elif id in np2_healthy_subjects:
            base_path = np2_process_path
        elif id in np5_healthy_subjects:
            base_path = np5_process_path
        elif id in np7_healthy_subjects:
            base_path = np7_process_path
        else:
            raise Exception(f'id: {id} not in any datasets.')
        path = convert_to_full_paths([id], base_path)
        test_set_subjects += path

    return test_set_subjects