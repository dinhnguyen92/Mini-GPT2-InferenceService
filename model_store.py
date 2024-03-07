import os
import json
import torch
import numpy as np

from models import Decoder, ModelConfig
from azure_blob_service import download_file, list_container_blobs

model_directory = 'model_dict'

models_container_name = 'ml-models'
model_file_prefix = 'mini-gpt2-'
state_dict_file_extension = 'pth'

config_container_name = 'ml-model-configs'
config_file_prefix = 'config-'
json_file_extension = 'json'

train_losses_container_name = 'ml-training-losses'
train_losses_file_prefix = 'train-losses-'

test_losses_container_name = 'ml-test-losses'
test_losses_file_prefix = 'test-losses-'

def download_model_config(model_version):
    file_name = f'{config_file_prefix}{model_version}.{json_file_extension}'
    with download_file(config_container_name, file_name) as buffer:
        return ModelConfig(**json.load(buffer))

def download_model(vocab_size, model_version, model_config: ModelConfig):
    file_name = f'{model_file_prefix}{model_version}.{state_dict_file_extension}'

    with download_file(models_container_name, file_name) as buffer:
        state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model = Decoder(vocab_size, model_config)
        model.load_state_dict(state_dict)
        return model
    
def download_losses(model_version, is_train_losses=True):
    prefix = train_losses_file_prefix if is_train_losses else test_losses_file_prefix
    container = train_losses_container_name if is_train_losses else test_losses_container_name
    file_name = f'{prefix}{model_version}.{json_file_extension}'
    with download_file(container, file_name) as buffer:
        return torch.tensor(np.array(json.load(buffer)))

def get_available_model_versions(limit=0):
    file_names = os.listdir(model_directory)
    versions = [name.removeprefix(model_file_prefix).removesuffix(f'.{state_dict_file_extension}') for name in file_names]

    # Sort the versions in ascending order
    sorted_versions = sorted(versions, key=lambda version: int(version[1:]))
    return sorted_versions if limit <= 0 else sorted_versions[-limit:]

def read_model_from_state_dict(vocab_size, model_version, model_config: ModelConfig):
    file_path = f'{model_directory}/{model_file_prefix}{model_version}.{state_dict_file_extension}'
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model = Decoder(vocab_size, model_config)
    model.load_state_dict(state_dict)
    return model