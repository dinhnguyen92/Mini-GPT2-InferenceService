import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from models import ModelInfo
from model_store import download_model_config, download_model, download_train_losses, get_available_model_versions

model_versions = get_available_model_versions()

class TextGenerator:
    def __init__(self):
        self.tokenizer_checkpoint = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_checkpoint)
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        self.model_infos = {version: ModelInfo(
            version=version,
            config=download_model_config(version),
            train_losses=download_train_losses(version)
        ) for version in model_versions}

        vocab_size = self.tokenizer.vocab_size + 1
        self.models = {version: download_model(vocab_size, self.model_infos[version]) for version in model_versions}

text_generator = TextGenerator()
app = FastAPI()

# Pydantic model for prompt request
class Prompt(BaseModel):
	text: str
	max_resp_len: int
	sampling_temp: float

# Pydantic model for generated text
class TextCompletion(BaseModel):
    tokens: List[str]

def temperature_sampling(logits, temperature=1.0):
    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax to get token probability distribution
    prob_dist = F.softmax(scaled_logits, dim=-1)

    # Sample token based on the probability distribution
    return torch.multinomial(prob_dist, 1)

def assert_valid_version(model_version):
    if model_version not in text_generator.models:
        raise HTTPException(status_code=404, detail=f"Model version '{model_version}' not found")
    
# Endpoint to get available model versions
@app.get('/model_versions', response_model=List[str])
async def list_versions():
    return model_versions

# Endpoint to get model's info
@app.get('/models/{model_version}/info', response_model=ModelInfo)
async def get_config(model_version: str):
    assert_valid_version(model_version)
    return text_generator.model_infos[model_version]

# Endpoint to generate text completion
@app.post('/models/{model_version}/generate', response_model=TextCompletion)
async def generate(model_version: str, prompt: Prompt):
    assert_valid_version(model_version)

    try:
        tokenized_prompt = text_generator.tokenizer(prompt.text, return_tensors='pt')
        input_ids = tokenized_prompt['input_ids']
        # Remove the last token from the input prompt if it is (SEP)
        if input_ids[:, -1].item() == text_generator.tokenizer.sep_token_id:
            input_ids = input_ids[:, :-1]

        for _ in range(prompt.max_resp_len):
            # Use the pre-trained model to generate the logits of the tokens
            output_logits = text_generator.models[model_version](input_ids)

            # We only use the last logit since we only want to predict the last/next token
            prediction_id = temperature_sampling(output_logits[:, -1, :], temperature=prompt.sampling_temp)
            input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))

            # Terminate the sequence if the token is (SEP)
            if prediction_id == text_generator.tokenizer.sep_token_id:
                break
        
        tokens = [text_generator.tokenizer.decode(input_id) for input_id in input_ids[0, :]]
        return {'tokens': tokens}
    
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f'Unexpected error: {ex}')

## Start the Server
if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
