import uvicorn
import logging
import traceback
from dotenv import dotenv_values

from typing import List

from fastapi import FastAPI, Response, HTTPException, Depends
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST
from auth import validate_api_key

from models import ModelInfo

from text_generator import TextGenerator, TextCompletion, Prompt
from loss_plot import generate_loss_plot


# Load environment variables
env_vars = dotenv_values('.env')
model_limit = int(env_vars['MODEL_LIMIT'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

text_generator = TextGenerator(logger, model_limit)
app = FastAPI(dependencies=[Depends(validate_api_key)])

def assert_valid_version(model_version):
    if model_version not in text_generator.models:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"Model version '{model_version}' not found")
    
@app.get('/health-check')
async def health_check():
    return {'message': 'success'}
    
# Endpoint to get available model versions
@app.get('/model-versions', response_model=List[str])
async def list_versions():
    try:
        return text_generator.model_versions
    except Exception as ex:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error getting model versions: {ex}')

# Endpoint to get model's info
@app.get('/models/{model_version}/info', response_model=ModelInfo)
async def get_model_info(model_version: str):
    assert_valid_version(model_version)
    try:
        return text_generator.model_infos[model_version]
    except Exception as ex:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error getting model info: {ex}')
    
# Endpoint to get model's loss plot
@app.get('/models/{model_version}/plot')
async def get_model_plot(model_version: str):
    assert_valid_version(model_version)
    try:
        train_losses = text_generator.model_infos[model_version].train_losses
        test_losses = text_generator.model_infos[model_version].test_losses
        plot_bytes = generate_loss_plot(train_losses, test_losses)
        return Response(content=plot_bytes, media_type="image/png")
    except Exception as ex:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error getting model plot: {ex}')

# Endpoint to generate text completion
@app.post('/models/{model_version}/generate', response_model=TextCompletion)
async def generate(model_version: str, prompt: Prompt):
    assert_valid_version(model_version)

    MIN_PROMPT_LENGTH = 3
    if len(prompt.text.split()) < MIN_PROMPT_LENGTH:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f'Prompt must have at least {MIN_PROMPT_LENGTH} words')

    try:
        return text_generator.generate_text(model_version, prompt)
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Error generating text: {ex}')

## Start the Server
if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
