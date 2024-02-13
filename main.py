import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from model_store import download_model, get_available_model_versions

# We need to import the model classes so that they can be unpickled from files
from models import CausalSelfAttention, TransformerBlock, PositionalEncoding, Decoder

model_versions = get_available_model_versions()

class TextGenerator:
    def __init__(self):
        self.models = {version: download_model(version) for version in model_versions}

        tokenizer_checkpoint = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_checkpoint)

		# Add pad token to the tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
          
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

# Endpoint to generate text completion
@app.post('/{model_version}/generate', response_model=TextCompletion)
async def generate(model_version: str, prompt: Prompt):
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
    
    tokens = []
    for input_id in input_ids[0, :]:
         tokens.append(text_generator.tokenizer.decode(input_id))

    return {'tokens': tokens}

## Start the Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
