import torch
import torch.nn as nn 

class InputProjection(nn.Module):
    def __init__(self, n_bins: int, d_model: int):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        self.linear = nn.Linear(n_bins, d_model)

    def forward(self, x):
        return self.linear(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, hop_length: int, sample_rate: int, dropout: float, scale: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Actual time per frame (in milliseconds)
        time_per_step = (hop_length / sample_rate) * 1000  # ms

        # (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) * time_per_step
        
        # (1, d_model // 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Create positional encodings
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = scale * torch.sin(position * div_term)
        pe[:, 1::2] = scale * torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model) for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)].detach()  # No grad required
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplier
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / (d_k ** 0.5)  # (batch_size, h, seq_len_q, seq_len_k)
        if mask is not None:
            attention_scores.masked_fill_(mask, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (batch_size, seq_len, d_model) 

        return self.linear_out(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout),
            ResidualConnection(dropout)
        ])

    def forward(self, x, src_key_padding_mask=None):
        # Convert padding mask to attention mask format if needed
        if src_key_padding_mask is not None:
            # Convert (batch_size, seq_len) to (batch_size, 1, 1, seq_len) for attention
            attention_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None
            
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, attention_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.norm(x)

    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, input_proj: InputProjection, encoder: Encoder, src_pos: TemporalPositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.input_proj = input_proj
        self.encoder = encoder
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_key_padding_mask=None):
        src = self.input_proj(src)  # Project 141 -> d_model
        src = self.src_pos(src)           # Add positional encoding
        return self.encoder(src, src_key_padding_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_seq_len: int, hop_length: int, sample_rate: int, d_model: int, num_classes: int, n_bins: int, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 0):
    if d_ff == 0:
        d_ff = 4 * d_model  # Default value for feed-forward dimension

    # create input projection layer
    input_proj = InputProjection(n_bins, d_model)

    # create positional encoding layers
    src_pos = TemporalPositionalEncoding(d_model, src_seq_len, hop_length, sample_rate, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionLayer(d_model, num_classes)

    transformer = Transformer(input_proj, encoder, src_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer