import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mamba_ssm import Mamba


class FeedForwardModule(nn.Module):
    """
    A standard full-step Feed-Forward Network with pre-LayerNorm.
    (Removed the 0.5 Macaron half-step multiplier).
    """
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.1):
        super(FeedForwardModule, self).__init__()
        hidden_dim = dim * expansion_factor
        
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.SiLU() 
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        # STANDARD RESIDUAL: No longer a Macaron half-step, so we remove the 0.5 multiplier
        return residual + x


class MambaSequenceModule(nn.Module):
    """
    The Bidirectional Mamba replacement for Multi-Head Self-Attention.
    Processes the sequence both forward and backward in time, concatenates 
    the results, and projects them back to the original dimension.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout_rate=0.1):
        super(MambaSequenceModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
        # We need two independent Mamba blocks to learn forward and backward patterns
        self.mamba_forward = Mamba(
            d_model=dim,      
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
        )
        
        self.mamba_backward = Mamba(
            d_model=dim,      
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
        )
        
        self.out_proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # 1. Forward Pass
        out_forward = self.mamba_forward(x)
        
        # 2. Backward Pass
        x_flipped = torch.flip(x, dims=[1])
        out_backward = self.mamba_backward(x_flipped)
        out_backward = torch.flip(out_backward, dims=[1])
        
        # 3. Concatenation and Projection
        out_concat = torch.cat([out_forward, out_backward], dim=2)
        x = self.out_proj(out_concat)
        x = self.dropout(x)
        
        return residual + x


class MambaformerBlock(nn.Module):
    """
    The Minimal Mambaformer Block: Mamba -> FFN -> LayerNorm
    (Removed heavy Conformer convolutions and Macaron topology).
    """
    def __init__(self, dim, ffn_expansion_factor=4, dropout_rate=0.1):
        super(MambaformerBlock, self).__init__()
        
        self.sequence_module = MambaSequenceModule(dim, dropout_rate=dropout_rate)
        self.ffn = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.sequence_module(x)
        x = self.ffn(x)
        x = self.final_layer_norm(x)
        return x


class ChordFormer(nn.Module):
    """
    The updated model architecture using Minimal Mambaformer blocks.
    """
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 num_layers: int, 
                 output_dims: List[int],
                 ffn_expansion_factor: int = 4, 
                 dropout_rate: float = 0.1):
        super(ChordFormer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Using the streamlined Minimal MambaformerBlock
        self.conformer_layers = nn.ModuleList([
            MambaformerBlock(
                dim=model_dim,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(model_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.input_projection(x)
        x = self.dropout(x)

        for layer in self.conformer_layers:
            x = layer(x)
        
        outputs = [head(x) for head in self.output_heads]
        return outputs


def build_chordformer(
    input_dim: int = 252,
    model_dim: int = 256,
    num_layers: int = 4,
    output_dims: Optional[List[int]] = None,
    ffn_expansion_factor: int = 4,
    dropout_rate: float = 0.1
) -> ChordFormer:
    """
    Builds, initializes, and returns the Minimal Mambaformer model.
    """
    if output_dims is None:
        output_dims = [
            12 * 7 + 1,  # (12 roots * 7 triad qualities) + 1 for 'No Chord'
            12 + 1,      # 12 bass notes + N
            4,           # N, 7, b7, bb7
            4,           # N, 9, #9, b9
            3,           # N, 11, #11
            3            # N, 13, b13
        ]

    model = ChordFormer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        output_dims=output_dims,
        ffn_expansion_factor=ffn_expansion_factor,
        dropout_rate=dropout_rate
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model


if __name__ == "__main__":
    chordformer_model = build_chordformer()
    print(chordformer_model)

    dummy_cqt = torch.randn(8, 1000, 252)
    predictions = chordformer_model(dummy_cqt)

    print("\n--- Test Run ---")
    print(f"Input shape: {dummy_cqt.shape}")
    print("Output shapes for each chord component head:")
    for i, p in enumerate(predictions):
        print(f"  Head {i+1}: {p.shape}")