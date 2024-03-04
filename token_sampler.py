import torch
import torch.nn.functional as F

from token_filter import is_valid_token

class TokenSampler:
    def __init__(self, tokenizer, logger):
        self.tokenizer = tokenizer
        self.logger = logger

    def temperature_sampling(self, logits, temperature):
        # Scale logits by temperature
        scaled_logits = logits / temperature

        # Apply softmax to get token probability distribution
        prob_dist = F.softmax(scaled_logits, dim=-1)

        # Sample token based on the probability distribution
        return torch.multinomial(prob_dist, 1)
    
    def sample_token_id(self, logits,  temperature=1.0):
        retry_count = 0
        max_retry = 200
        is_valid = False

        # Keep sampling until we get a valid token
        while retry_count < max_retry and not is_valid:
            # We only use the last logit since we only want to predict the last/next token
            token_id = self.temperature_sampling(logits[:, -1, :], temperature)
            token = self.tokenizer.decode(token_id[0, :])
            is_valid = is_valid_token(token)
            retry_count += 1

        if not is_valid:
            self.logger.info(f'Invalid character: {token}')

        return token_id