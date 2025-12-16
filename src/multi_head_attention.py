"""
Multi-head-attention class
"""
from torch import nn
from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_v=None):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_v if d_v is not None else self.d_k

        self.weights_q = nn.Linear(d_model, d_model, bias=False)
        self.weights_k = nn.Linear(d_model, d_model, bias=False)
        self.weights_v = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)

        self.spda = ScaledDotProductAttention(self.d_k)

        self.projection = nn.Linear(self.d_v * self.n_heads, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        q_shape: [B, T, D]
        """
        # Get q, k and v by applying linear layer
        q = self.weights_q(x)
        k = self.weights_k(x)
        v = self.weights_v(x)

        # Reshape to adding head dimension
        q = q.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)  # [B, T, H, d_k]
        k = k.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)  # [B, T, H, d_k]
        v = v.view(x.shape[0], x.shape[1], self.n_heads, self.d_v)  # [B, T, H, d_v]

        # Permute the H and T because of SDPA works on two last dimensions
        q = q.permute(0, 2, 1, 3)  # [B, H, T, d_k]
        k = k.permute(0, 2, 1, 3)  # [B, H, T, d_k]
        v = v.permute(0, 2, 1, 3)  # [B, H, T, d_v]

        # Broadcast mask with number of heads
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

        # Apply scaled-dot-product-attention
        attentions = self.spda(q, k, v, mask)  # [B, H, T, d_v]

        # Permute to reverse changes
        attentions = attentions.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.d_v * self.n_heads)  # [B, T, D]

        output = self.projection(attentions)  # [B, T, D]

        return output
