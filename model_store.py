import io
import pickle
import torch
from azure_blob_service import download_blob, list_container_blobs

models_container_name = 'ml-models'
model_file_prefix = 'mini_gpt2_'
modelf_file_suffix = '.pkl'

# Since the pickled model was trained with GPU but will now be loaded into CPU
# we need to create a custom unpickler to load it into CPU. More details below:
# https://github.com/pytorch/pytorch/issues/16797
class UnpicklerForCPU(pickle.Unpickler):
     def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def download_model(model_version):
    file_name = f'{model_file_prefix}{model_version}{modelf_file_suffix}'
    model_bytes = download_blob(models_container_name, file_name)
    with io.BytesIO(model_bytes) as model_file:
        return UnpicklerForCPU(model_file).load()
    
def get_available_model_versions():
    file_names = [blob.name for blob in list_container_blobs(models_container_name)]
    return [name.removeprefix(model_file_prefix).removesuffix(modelf_file_suffix) for name in file_names]