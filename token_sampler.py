import torch
import torch.nn.functional as F

from token_types import get_token_type, TokenType

class TokenSampler:
    def __init__(self, english_dictionary, tokenizer):
        self.english_dictionary = english_dictionary
        self.tokenizer = tokenizer

    def temperature_sampling(self, logits, temperature):
        # Scale logits by temperature
        scaled_logits = logits / temperature

        # Apply softmax to get token probability distribution
        prob_dist = F.softmax(scaled_logits, dim=-1)

        # Sample token based on the probability distribution
        return torch.multinomial(prob_dist, 1)
    
    def is_valid_pair(self, prev_token, curr_token):
        # 2 consecutive tokens are valid:
        # 1. If one is a punctuation, the other cannot be a punctuation
        # 2. If the second token is a suffix, the first one must be a prefix and both tokens combine to form a valid English word
        # 3. If the first token might be a prefix and the second token is not a suffix, the first token must be a valid English word
        # 4. For all other cases, neither of the token can be invalid
        prev_type = get_token_type(prev_token)
        curr_type = get_token_type(curr_token)
        if (prev_type == TokenType.PUNCTUATION and curr_type == TokenType.PUNCTUATION):
            return False
        elif curr_type == TokenType.SUFFIX:
            return prev_type == TokenType.MAYBE_PREFIX and self.is_english_word(prev_token + curr_token)
        elif prev_type == TokenType.MAYBE_PREFIX:
            return self.is_english_word(prev_type)
        else:
            return prev_type != TokenType.INVALID and curr_type != TokenType.INVALID
  
    def is_english_word(self, token):
        return get_token_type(token) == TokenType.MAYBE_PREFIX and token.lower().strip() in self.english_dictionary
	
    def are_valid_tokens(self, token_id_0, token_id_1, token_id_2):
        token_0 = self.tokenizer.decode(token_id_0[0, :])
        token_1 = self.tokenizer.decode(token_id_1[0, :])
        token_2 = self.tokenizer.decode(token_id_2[0, :])
        return self.is_valid_pair(token_0, token_1) and self.is_valid_pair(token_1, token_2)
  
    def sample_token_id(self, token_id_0, token_id_1, logits,  temperature=1.0):
        retry_count = 0
        max_retry = 100
        is_valid = False

        # Keep sampling until we get a valid token
        while retry_count < max_retry and not is_valid:
            # We only use the last logit since we only want to predict the last/next token
            token_id_2 = self.temperature_sampling(logits[:, -1, :], temperature)
            is_valid = self.are_valid_tokens(token_id_0, token_id_1, token_id_2)
            retry_count += 1

        return token_id_2