from typing import List
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from models import ModelInfo

from model_store import download_model_config, download_model, download_losses, get_available_model_versions
from token_sampler import TokenSampler
from token_filter import is_valid_token

def count_model_params(model):
    return sum(p.numel() for p in model.parameters())

# Pydantic model for prompt request
class Prompt(BaseModel):
	text: str
	max_resp_len: int
	sampling_temp: float

# Pydantic model for generated text
class TextCompletion(BaseModel):
    prompt_tokens: List[str]
    result_tokens: List[str]
    alt_token_groups: List[List[str]]
    alt_token_prob_groups: List[List[float]]

class TextGenerator:
    def __init__(self, logger, model_limit):
        self.logger = logger

        self.model_versions = get_available_model_versions(limit=model_limit)
        self.logger.info(f'All model versions: {self.model_versions}')

        self.tokenizer_checkpoint = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_checkpoint)
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        self.token_sampler = TokenSampler(self.tokenizer, logger)

        self.logger.info('Downloading model configs')
        self.model_configs = {version: download_model_config(version) for version in self.model_versions}

        self.logger.info('Downloading models')
        vocab_size = self.tokenizer.vocab_size + 1
        self.models = {version: download_model(
            vocab_size,
            version,
            self.model_configs[version]
        ) for version in self.model_versions}

        self.logger.info('Downloading model infos')
        self.model_infos = {version: ModelInfo(
            version=version,
            config=self.model_configs[version],
            test_losses=download_losses(version, is_train_losses=False),
            train_losses=download_losses(version),
            num_params=count_model_params(self.models[version])
        ) for version in self.model_versions}

        self.logger.info('Finished initializing TextGenerator')

    def decode_tokens(self, token_ids):
        return [self.tokenizer.decode(token_id) for token_id in token_ids]
    
    def get_top_k_alt_tokens(self, output_logits, temperature, k=5):
        # Using each token's distribution, we'll find the top k most probable tokens
        # that could serve as the original token's alternative
        prob_dist = self.token_sampler.get_temperature_scaled_prob_dist(output_logits, temperature)
        top_token_probs, top_token_ids = torch.topk(prob_dist, k=k, dim=2)
        
        # Convert the 3D token probs tensor to 2D list
        alt_token_prob_groups = top_token_probs.view(-1, top_token_probs.size(-1)).tolist()
        
        # Reduce the token tensor from 3D to 2D
        alt_token_id_groups = top_token_ids.view(-1, top_token_ids.size(-1))
        
        # Convert the token IDs into valid tokens
        alt_token_groups = []
        for alt_token_id_group in alt_token_id_groups:
            alt_tokens = self.decode_tokens(alt_token_id_group)
            valid_alt_tokens = [token for token in alt_tokens if is_valid_token(token)]
            alt_token_groups.append(valid_alt_tokens)
            
        return alt_token_groups, alt_token_prob_groups

    def generate_text(self, model_version: str, prompt: Prompt):
        tokenized_prompt = self.tokenizer(prompt.text, return_tensors='pt')
        input_ids = tokenized_prompt['input_ids']
        # Remove the last token from the input prompt if it is (SEP)
        if input_ids[:, -1].item() == self.tokenizer.sep_token_id:
            input_ids = input_ids[:, :-1]

        original_prompt_tokens = self.decode_tokens(input_ids[0, :])
        generated_token_probs = []

        for _ in range(prompt.max_resp_len):
            output_logits = self.models[model_version](input_ids)
            prediction_id, id_prob = self.token_sampler.sample_token_id(output_logits, temperature=prompt.sampling_temp)

            # Insert the prediction ID back into the input sequence so that
            # it will become part of the inputs to predict the next token
            input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))
            generated_token_probs.append(id_prob)
            
            # Terminate the sequence if the token is (SEP)
            if prediction_id == self.tokenizer.sep_token_id:
                break
        
        # The final output_logits tensor contains the probability distribution of all generated tokens
        # Hence we can use it to find alternative tokens for all final generated tokens
        alt_token_groups, alt_token_prob_groups = self.get_top_k_alt_tokens(output_logits, temperature=prompt.sampling_temp, k=5)
        result_tokens = self.decode_tokens(input_ids[0, :])

        return TextCompletion(
            prompt_tokens=original_prompt_tokens,
            result_tokens=result_tokens,
            alt_token_groups=alt_token_groups,
            alt_token_prob_groups=alt_token_prob_groups
        )
