import math
from typing import List
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig(BaseModel):
    batch_size: int
    max_len: int
    d_k: int
    d_model: int
    n_heads: int
    n_layers: int
    epochs: int
    dropout_rate: float
    lr: float
    lr_min: float
    T_0: int
    T_mult: int

class ModelInfo(BaseModel):
    version: str
    config: ModelConfig
    test_losses: List[float]
    train_losses: List[float]
    num_params: int

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CausalSelfAttention, self).__init__()

        # Assume d_v = d_k
        self.d_k = config.d_k
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.key = nn.Linear(config.d_model, config.n_heads * config.d_k)
        self.query = nn.Linear(config.d_model, config.n_heads * config.d_k)
        self.value = nn.Linear(config.d_model, config.n_heads * config.d_k)

        causal_mask = torch.tril(torch.ones(config.max_len, config.max_len))
        # Save and reshape the causal mask tensor to be 4D so that it can be
        # broadcasted along the sample dimension (N) and time-series dimension (T)
        self.register_buffer(
            'causal_mask',
            causal_mask.view(1, 1, config.max_len, config.max_len)
        )

        # Final linear classification layer
        self.fc = nn.Linear(config.n_heads * config.d_k, config.d_model)

    def forward(self, q, k, v):
        q = self.query(q) # N x T x (n_heads * d_k)
        k = self.key(k) # N x T x (n_heads * d_k)
        v = self.value(v) # N x T x (n_heads * d_k)

        N = q.shape[0]
        T_output = q.shape[1]
        T_input = k.shape[1]

        def reshape(tensor, T):
            # Change shape to:
            # (N, T, n_heads, d_k) => (N, n_heads, T, d_k)
            # in order for mat-mul to work properly
            return tensor.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        q = reshape(q, T_output)
        k = reshape(k, T_input)
        v = reshape(v, T_input)

        # We'll compute the attention matrix based on this formula:
        # Attention(Q, K, V) = softmax((Q x K-transposed)/(squared root of d_k)) x V

        # Step 1: compute attention scores (Q x K-transposed)/(squared root of d_k)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # Step 2: Apply causal mask
        # To prevent the network to pay attention to a particular concept,
        # set its corresponding attention score to negative infinity
        # so that when the softmax function is applied, the mask will yield zero.
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T_output, :T_input] == 0, float('-inf'))

        # Step 3: apply softmax to convert the attention scores into attention probabilities/weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Step 4: Compute the attention-weighted values, i.e. the attenion matrix
        A = attn_weights @ v

        # Reshape the attention matrix before multiplying it with final linear layer:
        A = A.transpose(1, 2)
        A = A.contiguous().view(N, T_output, self.n_heads * self.d_k)

        # Project A onto the final linear classification layer.
        return self.fc(A)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerBlock, self).__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(p=config.dropout_rate)
        self.dropout2 = nn.Dropout(p=config.dropout_rate)

        self.csa = CausalSelfAttention(config)
        self.ann = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

    def forward(self, x):

        norm_input = self.ln1(x)
        csa_output = self.csa(norm_input, norm_input, norm_input)
        res_output1 = x + self.dropout1(csa_output)

        norm_res_output1 = self.ln2(res_output1)
        ann_output = self.ann(norm_res_output1)
        res_output2 = norm_res_output1 + self.dropout2(ann_output)

        return res_output2

class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # arange(max_len) creates a 1D tensor with elements ranging from 0 to max_len - 1
        # unsqueeze(1) then adds an extra dimension to this tensor along dimension 1, i.e. along the column.
        # The end result will be a 2D tensor with max_len rows and 1 column.
        position = torch.arange(config.max_len).unsqueeze(1)

        # arange(0, d_model, 2) creates a 1D tensor with elements starting from 0
        # and incrementing by 2 until it reaches a value less than d_model.
        exp_term = torch.arange(0, config.d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(1000.0) / config.d_model))

        pe = torch.zeros(1, config.max_len, config.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # Save the positional encoding
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Shape of X: N x T x d_model
        # T is the number of tokens within a sample/time-series
        # T can only be at most equal to max_len
        # Here we use :x.size(1) to specify the number of pre-computed position encodings (T) we'll use
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, config: ModelConfig):
        super(Decoder, self).__init__()

        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)

        transformer_blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.ln = nn.LayerNorm(config.d_model)
        # Since the decoder is used for predicting the next token in the series
        # the number of possible output "classes" is equal to the vocab size
        self.fc = nn.Linear(config.d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln(x)
        x = self.fc(x)
        return x