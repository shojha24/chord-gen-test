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
    
class PositionWiseConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, n_conv_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        conv_layers = []
        for _ in range(n_conv_layers):
            conv_layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Conv1d expects: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        return x.transpose(1, 2)  # Back to (batch_size, seq_len, d_model)
    
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
            attention_scores.masked_fill_(mask == 0, float('-inf'))
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
    
class BiDirectionalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        self.forward_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.backward_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.output_projection = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x, input_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create causal masks
        forward_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        backward_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                
        # Apply attention in both directions
        forward_out = self.forward_attention(x, x, x, forward_mask)
        backward_out = self.backward_attention(x, x, x, backward_mask)
        
        # Concatenate and project
        combined = torch.cat([forward_out, backward_out], dim=-1)
        return self.output_projection(combined)

    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class BTCEncoderBlock(nn.Module):
    def __init__(self, bi_attention_block: BiDirectionalAttentionBlock, 
                 conv_block: PositionWiseConvBlock, dropout: float):
        super().__init__()
        self.bi_attention_block = bi_attention_block
        self.conv_block = conv_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout),
            ResidualConnection(dropout)
        ])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.bi_attention_block(x))
        x = self.residual_connections[1](x, self.conv_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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

    def encode(self, src):
        src = self.input_proj(src)  # Project 141 -> d_model
        src = self.src_pos(src)           # Add positional encoding
        return self.encoder(src)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_seq_len: int, hop_length: int, sample_rate: int, d_model: int, num_classes: int, n_bins: int, N: int = 6, h: int = 8, dropout: float = 0.1, n_conv_layers: int = 2,  kernel_size: int = 3):

    # create input projection layer
    input_proj = InputProjection(n_bins, d_model)

    # create positional encoding layers
    src_pos = TemporalPositionalEncoding(d_model, src_seq_len, hop_length, sample_rate, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        bi_attention_block = BiDirectionalAttentionBlock(d_model, h, dropout)
        conv_block = PositionWiseConvBlock(d_model, kernel_size, n_conv_layers, dropout)
        encoder_block = BTCEncoderBlock(bi_attention_block, conv_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionLayer(d_model, num_classes)

    transformer = Transformer(input_proj, encoder, src_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer