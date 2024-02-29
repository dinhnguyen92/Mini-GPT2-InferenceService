import io
from dotenv import dotenv_values
from azure.storage.blob import BlobServiceClient

# Load environment variables
env_vars = dotenv_values('.env')

# We need a large connection timeout since large models take a long time to download
CONNECTION_TIMEOUT_SEC = 600

def get_container_client(container_name: str):
    service_client = BlobServiceClient.from_connection_string(env_vars['AZURE_BLOB_STORAGE_CONNECTION_STRING'])
    return service_client.get_container_client(container_name)

def download_file(container_name: str, file_name: str):
    blob_client = get_container_client(container_name).get_blob_client(file_name)
    blob_bytes = blob_client.download_blob(connection_timeout=CONNECTION_TIMEOUT_SEC).readall()
    return io.BytesIO(blob_bytes)
    
def list_container_blobs(container_name: str):
    container_client = get_container_client(container_name)
    return container_client.list_blobs()