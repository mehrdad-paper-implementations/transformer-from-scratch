"""
Scaled Dot Product Attention class
"""
import math
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        """
        q_shape: [B, T, d_k] or [B, M, T, d_k]
        k_shape: [B, T, d_k] or [B, M, T, d_k]
        v_shape: [B, T, d_v] or [B, M, T, d_v]
        """
        k_t = torch.transpose(k, -2, -1)
        attention_score = torch.matmul(q, k_t)
        attention_score /= math.sqrt(self.d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

        attention_score = self.softmax(attention_score)
        attention = torch.matmul(attention_score, v)

        return attention
