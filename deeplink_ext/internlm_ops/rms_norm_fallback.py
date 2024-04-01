# Copyright (c) 2024, DeepLink.

import torch

__all__ = ["RMSNorm", "RMSNormWithNormalizedShape"]


# RMSNorm fallback from InternLM
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        InternLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


RMSNormWithNormalizedShape = RMSNorm
