import uvicorn
import io
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from dotenv import dotenv_values

import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from models import CausalSelfAttention, TransformerBlock, PositionalEncoding, Decoder

# Load environment variables
env_vars = dotenv_values(".env")

# Since the pickled model was trained with GPU but will now be loaded into CPU
# we need to create a custom unpickler to load it into CPU. More details below:
# https://github.com/pytorch/pytorch/issues/16797
class UnpicklerForCPU(pickle.Unpickler):
     def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class TextGenerator:
    def __init__(self, model_path):
        with open(model_path, mode='rb') as file:
            self.model = UnpicklerForCPU(file).load()
						
        tokenizer_checkpoint = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_checkpoint)

		# Add pad token to the tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
          
text_generator = TextGenerator(env_vars['MODEL_PATH'])
app = FastAPI()

# Pydantic model for prompt request
class Prompt(BaseModel):
	text: str
	max_resp_len: int
	sampling_temp: float

# Pydantic model for generated text
class GeneratedCompletion(BaseModel):
    tokens: List[str]

def temperature_sampling(logits, temperature=1.0):
    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax to get token probability distribution
    prob_dist = F.softmax(scaled_logits, dim=-1)

    # Sample token based on the probability distribution
    return torch.multinomial(prob_dist, 1)

@app.post('/generate', response_model=GeneratedCompletion)
async def generate(prompt: Prompt):
    tokenized_prompt = text_generator.tokenizer(prompt.text, return_tensors='pt')

    input_ids = tokenized_prompt['input_ids']
    # Remove the last token from the input prompt if it is (SEP)
    if input_ids[:, -1].item() == text_generator.tokenizer.sep_token_id:
        input_ids = input_ids[:, :-1]

    for _ in range(prompt.max_resp_len):
        # Use the pre-trained model to generate the logits of the tokens
        output_logits = text_generator.model(input_ids)

        # We only use the last logit since we only want to predict the last/next token
        prediction_id = temperature_sampling(output_logits[:, -1, :], temperature=prompt.sampling_temp)
        input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))

        # Terminate the sequence if the token is (SEP)
        if prediction_id == text_generator.tokenizer.sep_token_id:
            break
    
    tokens = []
    for input_id in input_ids[0, :]:
         tokens.append(text_generator.tokenizer.decode(input_id))

    return { 'tokens': tokens }

## Start the Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
