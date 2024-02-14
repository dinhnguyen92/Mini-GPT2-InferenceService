import json
import torch
import numpy as np

from models import Decoder, ModelConfig, ModelInfo
from azure_blob_service import download_file, list_container_blobs

models_container_name = 'ml-models'
model_file_prefix = 'mini-gpt2-'
state_dict_file_extension = 'pth'

config_container_name = 'ml-model-configs'
config_file_prefix = 'config-'
json_file_extension = 'json'

train_losses_container_name = 'ml-training-losses'
train_losses_file_prefix = 'train-losses-'

def download_model_config(model_version):
    file_name = f'{config_file_prefix}{model_version}.{json_file_extension}'
    with download_file(config_container_name, file_name) as buffer:
        return ModelConfig(**json.load(buffer))

def download_model(vocab_size, model_info: ModelInfo):
    file_name = f'{model_file_prefix}{model_info.version}.{state_dict_file_extension}'

    with download_file(models_container_name, file_name) as buffer:
        state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model = Decoder(vocab_size, model_info.config)
        model.load_state_dict(state_dict)
        return model

def download_train_losses(model_version):
    file_name = f'{train_losses_file_prefix}{model_version}.{json_file_extension}'
    with download_file(train_losses_container_name, file_name) as buffer:
        return torch.tensor(np.array(json.load(buffer)))

def get_available_model_versions():
    file_names = [blob.name for blob in list_container_blobs(models_container_name)]
    return [name.removeprefix(model_file_prefix).removesuffix(f'.{state_dict_file_extension}') for name in file_names]