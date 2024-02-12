from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import dotenv_values
from contextlib import asynccontextmanager

import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

class TextGenerator:
	def __init__(self):
		self.model = None
		self.tokenizer = None
          
text_generator = TextGenerator()

# Load environment variables
env_vars = dotenv_values(".env")

# This will be executed when the applicatin starts up
@asynccontextmanager
async def lifespan(application: FastAPI):
	# Load the Mini GPT-2 model
	if text_generator.model is None:
		file_path = env_vars['MODEL_PATH']
		with open(file_path, 'rb') as file:
			text_generator.model = pickle.load(file)

	# Load the tokenizer
	if text_generator.tokenizer is None:
		tokenizer_checkpoint = 'gpt2'
		tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_checkpoint)

		# Add pad token to the tokenizer
		tokenizer.add_special_tokens({"pad_token": "<PAD>"})

app = FastAPI(lifespan=lifespan)

# Pydantic model for prompt request
class Prompt(BaseModel):
  text: str
  max_resp_len: int
  sampling_temp: float

# Pydantic model for generated text
class GeneratedCompletion(BaseModel):
  text: str

def temperature_sampling(logits, temperature=1.0):
    # Scale input logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax to get token probability distribution
    prob_dist = F.softmax(scaled_logits, dim=-1)

    # Sample token based on the probability distribution
    return torch.multinomial(prob_dist, 1)

@app.post('/generate', response_model=GeneratedCompletion)
async def generate(prompt: Prompt):
    tokenized_prompt = text_generator.tokenizer(prompt.text, return_tensors='pt')

    # Remove the last token (SEP) from the input prompt
    input_ids = tokenized_prompt['input_ids'][:, :-1]

    for _ in range(prompt.max_resp_len):
        output_logits = text_generator.model(input_ids)
        # We only use the last logit since we only want to predict the last/next token
        prediction_id = temperature_sampling(output_logits[:, -1, :], temperature=prompt.sampling_temp)
        input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))

        if prediction_id == text_generator.tokenizer.sep_token_id:
            break

    generated = text_generator.tokenizer.decode(input_ids[0, 1:])
    return {'text': generated}
