from azure.storage.blob import BlobServiceClient
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values('.env')

def get_container_client(container_name: str):
    service_client = BlobServiceClient.from_connection_string(env_vars['AZURE_BLOB_CONNECTION_STRING'])
    return service_client.get_container_client(container_name)

def download_blob(container_name: str, file_name: str):
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(file_name)
    return blob_client.download_blob().readall()
    
def list_container_blobs(container_name: str):
    container_client = get_container_client(container_name)
    return container_client.list_blobs()