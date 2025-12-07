import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, n_head: int, dropout_rate: float = 0.3):
        super().__init__()

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.n_head = n_head
        self.head_dim = query_dim // n_head

        if self.head_dim * n_head != query_dim:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by n_head ({n_head})")

        self.scale = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate) # Added dropout

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query_input: torch.Tensor, kv_input: torch.Tensor, attn_mask = None) -> torch.Tensor:
        batch_size, query_seq_len, _ = query_input.shape # query_seq_len will be 1
        _, kv_seq_len, _ = kv_input.shape

        q = self.q_proj(query_input) # (batch, query_seq_len, query_dim)
        k = self.k_proj(kv_input)    # (batch, kv_seq_len, query_dim)
        v = self.v_proj(kv_input)    # (batch, kv_seq_len, query_dim)

        Q = q.view(batch_size, query_seq_len, self.n_head, self.head_dim).transpose(1, 2) # (batch, n_head, query_seq_len, head_dim)
        K = k.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)    # (batch, n_head, kv_seq_len, head_dim)
        V = v.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)    # (batch, n_head, kv_seq_len, head_dim)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale # (batch, n_head, query_seq_len, kv_seq_len)
       
        # Implementing mask on the score for decoder transformers
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights) # Added dropout after softmax

        attn_output = attn_weights @ V # (batch, n_head, query_seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous() # (batch, query_seq_len, n_head, head_dim)
        attn_output = attn_output.view(batch_size, query_seq_len, self.query_dim) # (batch, query_seq_len, query_dim)

        output = self.out_proj(attn_output) # (batch, query_seq_len, query_dim)
        output = self.dropout(output) # Added dropout after output projection

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, linear_units, dropout_rate):
        super().__init__()
        self.w_1 = nn.Linear(d_model, linear_units)
        self.w_2 = nn.Linear(linear_units, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.w_2(F.relu(self.w_1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.norm3 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt_norm = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask))

        # Cross-attention
        tgt_norm = self.norm2(tgt)
        memory_norm = self.norm2(memory) # Apply norm to memory as well
        tgt = tgt + self.dropout(self.src_attn(tgt_norm, memory_norm, memory_norm, attn_mask=memory_mask))

        # Feed-forward
        tgt_norm = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt_norm))
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Logic moved from BaseTransformerDecoder
        self.embed = nn.Embedding(config.vocab_size, config.encoder_output_size) # Assuming embed_size == encoder_output_size for simplicity
        self.pos_enc = pos_enc_class(config.encoder_output_size, config.positional_dropout_rate) if config.pos_enc_class else None
        self.output_layer = nn.Linear(config.encoder_output_size, config.vocab_size) if config.use_output_layer else None

        attention_dim = config.encoder_output_size
        self.decoders = nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadAttention(
                    query_dim=attention_dim, kv_dim=attention_dim, n_head=config.attention_heads, dropout_rate=config.self_attention_dropout_rate
                ),
                MultiHeadAttention(
                    query_dim=attention_dim, kv_dim=attention_dim, n_head=config.attention_heads, dropout_rate=config.src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, config.linear_units, config.dropout_rate),
                config.dropout_rate,
            ) for _ in range(config.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor, # This is our concept_vector
        memory_mask: Optional[torch.Tensor],
        ys_in_pad: torch.Tensor, # Target sequence (captions_input_ids)
        ys_in_lens: torch.Tensor, # Lengths of target sequence
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tgt = ys_in_pad
        # tgt_mask: (batch, 1, maxlen_out)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen=tgt.size(1)).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).unsqueeze(0).to(tgt.device)

        x = self.embed(tgt)
        if self.pos_enc:
            x = self.pos_enc(x)

        for layer in self.decoders:
            x = layer(x, memory, tgt_mask, memory_mask)

        if self.output_layer:
            x = self.output_layer(x)

        return x, None, None # Returning logits, and None for attention weights for simplicity

    def forward_one_step(
        self,
        tgt: torch.Tensor, # (batch, current_length)
        memory: torch.Tensor, # (batch, 1, concept_dim)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(tgt)
        if self.pos_enc:
            x = self.pos_enc(x)

        for layer in self.decoders:
            x = layer(x, memory, tgt_mask, memory_mask)

        if self.output_layer:
            x = self.output_layer(x[:, -1]) # Only predict the last token

        return x, None # Returning logits, and None for attention weights for simplicity


def make_pad_mask(lengths: torch.Tensor, maxlen: Optional[int] = None) -> torch.Tensor:
    if maxlen is None:
        maxlen = lengths.max().item()
    seq_range = torch.arange(maxlen, device=lengths.device)
    x = seq_range.unsqueeze(0) < lengths.unsqueeze(1)
    return x


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
