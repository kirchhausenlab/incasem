import os
import json

def create_multiple_config(config_file):
    '''
    From one config .json file that contains multiple datasets,
    create  multiple .json files, each one containing a single dataset.
    Used to perform validation/prediction on multiple cells/datasets.
    
    Arguments:
        config_file (str) : path of the config file
    
    Returns:
        datasets (dict) : dictionary of created config files {path, number}

    '''
    with open(os.path.expanduser(config_file), 'r') as f:
        config = json.load(f)

        datasets = []
        for idx, key in enumerate(config):
            numbered_path = os.path.expanduser(config_file).replace('.json', f'_{idx}.json')

            path_list = numbered_path.split('/')

            temp_folder = os.path.join(
                '/'.join(path_list[:-1]),
                'temp')

            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)

            temp_path = os.path.join(
                temp_folder,
                path_list[-1]
            )
            datasets.append((temp_path, key))

            with open(datasets[-1][0], 'w') as outfile:
                json.dump({key: config[key]}, outfile)

        return datasets

