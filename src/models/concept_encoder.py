import torch
import torch.nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int = 512, nhead: int = 4):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.scale = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)


        self.out_proj = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, concepts: torch.Tensor, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        # TODO:: I think a good idea here is to have a method to dropout certain modalities representation for some concepts
        # I think this can make the model more robust and make it less depedent on one modality

        batch_size, img_seq_len, hidden_dim = images.shape
        _, txt_seq_len, _ = texts.shape
        kv_seq_len = img_seq_len + txt_seq_len

        concepts = concepts.unsqueeze(1) # reshape to: (batch_size, 1, hidden_dim)
        kv = torch.concat([images, texts], dim=1) # (batch_size, img_seq_len + txt_seq_len, hidden_dim)

        q = self.q_proj(concepts)
        k = self.k_proj(kv)
        v = self.v_proj(kv)


        Q = q.view(batch_size, 1, self.nhead, self.head_dim).transpose(1, 2)
        K = k.view(batch_size, kv_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = v.view(batch_size, kv_seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = self.softmax(attn_scores)

        attn_output = attn_weights @ V

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, kv_seq_len, hidden_dim)


        output = self.out_proj(attn_output)

        return output

