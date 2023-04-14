from copy import deepcopy

def convert_keys(old_dict):
    new_state_dict = {}
    for key in old_dict:
        new_key = key.split('model.')[1]
        new_state_dict[new_key] = deepcopy(old_dict[key])
    return new_state_dict