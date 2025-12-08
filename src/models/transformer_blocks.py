import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class SlotAttention(nn.Module):
    def __init__(self, slot_dim: int, num_iter: int = 3, num_slots: int = 10):
        super().__init__()

        self.num_iter = num_iter
        self.num_slots = num_slots

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

        # Defining the mu, sigma parameters for initializing the slots from
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.constant_(self.slot_log_sigma, -1.0)

        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)


        # Defining epsilon for numerical stability
        self.epsilon = 1e-7

        # NOTE: Here we do the softmax around dim = 1 which is the slots dim as in the paper: https://arxiv.org/pdf/2006.15055
        self.softmax = nn.Softmax(dim=1)

        # NOTE: This layer is used for updating the slots smoothly
        self.gru_cell = nn.GRUCell(slot_dim, slot_dim)

    def generate_slots(self, batch_size: int) -> torch.Tensor:
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)

        slots = mu + sigma * torch.randn_like(mu)
        return slots

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq, dim = features.shape

        features = self.norm1(features)

        slots = self.generate_slots(batch_size)

        K = self.k_proj(features) # (batch_size, seq, dim)
        V = self.v_proj(features) # (batch_size, seq, dim)

        for t in range(self.num_iter):
            slots_prev = slots
            slots = self.norm2(slots)

            Q = self.q_proj(slots) # (batch_size, num_slots, dim)

            scores = Q @ K.transpose(-2, -1) # (batch_size, num_slots, seq)

            attn = self.softmax(scores) + self.epsilon

            updates = attn @ V # (batch_size, num_slots, dim)

            slots = self.gru_cell(slots_prev, updates)

        return slots

