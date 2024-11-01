import torch
import torch.nn.functional as F

class AttentionScaling:
    def __init__(self, ntk_scale, pi_window_size):
        self.ntk_scale = ntk_scale
        self.pi_window_size = pi_window_size

    def ntk_rope(self, embeddings):
        # Apply NTK-RoPE scaling on the embeddings
        scaling_factor = torch.exp(torch.arange(0, embeddings.size(-1), 2).float() * -self.ntk_scale).to(embeddings.device)
        scaled_embeddings = embeddings * scaling_factor
        return scaled_embeddings

    def position_interpolation(self, embeddings):
        # Position interpolation for attention scaling
        sequence_length = embeddings.size(1)
        if sequence_length != self.pi_window_size:
            return F.interpolate(embeddings, size=self.pi_window_size, mode='linear')
        return embeddings
    