import torch
import torch.nn.functional as F

from token_filter import is_valid_token

class TokenSampler:
    def __init__(self, tokenizer, logger):
        self.tokenizer = tokenizer
        self.logger = logger

    def get_temperature_scaled_prob_dist(self, logits, temperature):
        # Scale logits by temperature
        scaled_logits = logits / temperature

        # Apply softmax to get token probability distribution
        return F.softmax(scaled_logits, dim=-1)

    def temperature_sampling(self, logits, temperature):
        prob_dist = self.get_temperature_scaled_prob_dist(logits, temperature)

        # Sample token based on the probability distribution
        token_id = torch.multinomial(prob_dist, 1)
        token_prob = prob_dist[0, token_id].item()
        return (token_id, token_prob)
    
    def sample_token_id(self, logits,  temperature=1.0):
        retry_count = 0
        max_retry = 200
        is_valid = False

        # Keep sampling until we get a valid token
        while retry_count < max_retry and not is_valid:
            # We only use the last logit since we only want to predict the last/next token
            token_id, token_prob = self.temperature_sampling(logits[:, -1, :], temperature)
            token = self.tokenizer.decode(token_id[0, :])
            is_valid = is_valid_token(token)
            retry_count += 1

        if not is_valid:
            self.logger.info(f'Invalid character: {token}')

        return token_id, token_prob