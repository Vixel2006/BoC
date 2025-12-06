import torch
import torch.nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, n_head: int):
        super().__init__()

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.n_head = n_head
        self.head_dim = query_dim // n_head

        if self.head_dim * n_head != query_dim:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by n_head ({n_head})")

        self.scale = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        batch_size, query_seq_len, _ = query_input.shape # query_seq_len will be 1
        _, kv_seq_len, _ = kv_input.shape

        q = self.q_proj(query_input) # (batch, query_seq_len, query_dim)
        k = self.k_proj(kv_input)    # (batch, kv_seq_len, query_dim)
        v = self.v_proj(kv_input)    # (batch, kv_seq_len, query_dim)

        Q = q.view(batch_size, query_seq_len, self.n_head, self.head_dim).transpose(1, 2) # (batch, n_head, query_seq_len, head_dim)
        K = k.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)    # (batch, n_head, kv_seq_len, head_dim)
        V = v.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)    # (batch, n_head, kv_seq_len, head_dim)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale # (batch, n_head, query_seq_len, kv_seq_len)
        attn_weights = self.softmax(attn_scores)

        attn_output = attn_weights @ V # (batch, n_head, query_seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous() # (batch, query_seq_len, n_head, head_dim)
        attn_output = attn_output.view(batch_size, query_seq_len, self.query_dim) # (batch, query_seq_len, query_dim)

        output = self.out_proj(attn_output) # (batch, query_seq_len, query_dim)

        return output

