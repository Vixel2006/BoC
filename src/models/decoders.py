from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .transformer_blocks import *
from ...configs.image_decoder_config import ImageDecoderConfig
from ...configs.text_decoder_config import TextDecoderConfig
import math

class SlotImageDecoder(nn.Module):
    def __init__(self, config: ImageDecoderConfig):
        super().__init__()
        self.config = config
        
        # Calculate initial size
        self.initial_dim = config.image_size // (2 ** (config.num_decoder_layers - 1))
        
        # Input: One single concept vector
        self.fc = nn.Linear(config.input_dim, config.hidden_dim * self.initial_dim * self.initial_dim)
        
        layers = []
        in_channels = config.hidden_dim
        
        for i in range(config.num_decoder_layers):
            # Last layer outputs 4 channels: 3 for RGB, 1 for Alpha Mask
            is_last = (i == config.num_decoder_layers - 1)
            out_channels = 4 if is_last else in_channels // 2
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4, stride=2, padding=1
                )
            )
            if not is_last:
                layers.append(nn.ReLU())
            
            in_channels = out_channels
            
        self.decoder = nn.Sequential(*layers)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        # Input concepts: (Batch_Size, Num_Slots, Input_Dim)
        B, N, D = concepts.shape
        
        # 1. FLATTEN: Treat every slot as an independent sample
        # Shape becomes: (Batch_Size * Num_Slots, Input_Dim)
        flat_concepts = concepts.reshape(B * N, D)
        
        # 2. Decode features
        x = self.fc(flat_concepts)
        x = x.view(B * N, self.config.hidden_dim, self.initial_dim, self.initial_dim)
        x = self.decoder(x) # Output: (B*N, 4, H, W)
        
        # 3. UNFLATTEN: Separate back into batches and slots
        H, W = x.shape[2], x.shape[3]
        x = x.view(B, N, 4, H, W)
        
        # 4. Split RGB and Alpha
        rgb = x[:, :, :3, :, :]   # (B, N, 3, H, W)
        alpha = x[:, :, 3:4, :, :] # (B, N, 1, H, W)
        
        # 5. Recombine (Composition)
        # We enforce alpha to sum to 1 across slots using Softmax
        alpha = F.softmax(alpha, dim=1)

        # Final Image = Sum(RGB * Alpha)
        recon_image = torch.sum(rgb * alpha, dim=1) # (B, 3, H, W)
        
        return recon_image, alpha # Return alpha for visualization!

class SlotTextDecoderGRU(nn.Module):
    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, num_layers=config.num_decoder_layers, batch_first=True)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Project combined concepts to hidden state
        self.concept_to_hidden = nn.Linear(config.input_dim, config.hidden_dim * config.num_decoder_layers)

    def forward(self, concepts: torch.Tensor, target_sequence=None):
        # concepts: (Batch, Num_Slots, Dim)
        B, N, D = concepts.shape
        
        global_context = torch.mean(concepts, dim=1) # (Batch, Dim)
        
        # Now proceed as normal
        h_0 = self.concept_to_hidden(global_context)
        h_0 = h_0.view(self.config.num_decoder_layers, B, self.config.hidden_dim)


        if target_sequence is not None:
            # Teacher forcing during training
            embedded = self.embedding(target_sequence) # (batch_size, seq_len, hidden_dim)
            output, _ = self.gru(embedded, h_0) # output: (batch_size, seq_len, hidden_dim)
            logits = self.fc_out(output) # (batch_size, seq_len, vocab_size)
            return logits
        else:
            # Inference mode: generate sequence token by token
            # Start with a special <SOS> token (assuming token 0 is <SOS>)
            input_token = torch.zeros((B, 1), dtype=torch.long, device=concepts.device) # Use B for batch_size
            
            outputs = []
            hidden = h_0
            
            for _ in range(self.config.max_seq_len):
                embedded = self.embedding(input_token) # (batch_size, 1, hidden_dim)
                output, hidden = self.gru(embedded, hidden) # output: (batch_size, 1, hidden_dim)
                logits = self.fc_out(output.squeeze(1)) # (batch_size, vocab_size)
                
                outputs.append(logits)
                
                # Get the next input token (greedy sampling)
                input_token = logits.argmax(1).unsqueeze(1)

            
            return torch.stack(outputs, dim=1) # (batch_size, max_seq_len, vocab_size)
