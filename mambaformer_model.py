import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mamba_ssm import Mamba


class FeedForwardModule(nn.Module):
    """
    The 'half-step' feed-forward module (FFN) with pre-LayerNorm.
    """
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.1):
        super(FeedForwardModule, self).__init__()
        hidden_dim = dim * expansion_factor
        
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.SiLU() # Using the efficient, built-in Swish
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
        # Apply the 0.5 scaling for the residual connection
        return residual + 0.5 * x


class MambaSequenceModule(nn.Module):
    """
    The Bidirectional Mamba replacement for Multi-Head Self-Attention.
    Processes the sequence both forward and backward in time, concatenates 
    the results, and projects them back to the original dimension.
    Retains the pre-LayerNorm and residual connection structure to match 
    the original Conformer block design.
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
        
        # Because we concatenate the two outputs (dim + dim = 2*dim), 
        # we need a linear layer to project it back down to the model dimension
        self.out_proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # 1. Forward Pass
        out_forward = self.mamba_forward(x)
        
        # 2. Backward Pass
        # We flip the sequence along the time dimension (dim=1)
        x_flipped = torch.flip(x, dims=[1])
        out_backward = self.mamba_backward(x_flipped)
        # Flip the output back to normal chronological order
        out_backward = torch.flip(out_backward, dims=[1])
        
        # 3. Concatenation and Projection
        # Concatenate along the feature dimension (dim=2)
        out_concat = torch.cat([out_forward, out_backward], dim=2)
        
        # Project back down to the original model dimension
        x = self.out_proj(out_concat)
        
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    The convolution module with pre-LayerNorm, GLU, and depthwise convolution.
    """
    def __init__(self, dim, kernel_size=31, dropout_rate=0.1):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # Transpose for Conv1d which expects (batch, channels, length)
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back to (batch, length, channels)
        x = x.transpose(1, 2)
        
        return residual + x


class MambaformerBlock(nn.Module):
    """
    The updated block: FFN -> Mamba -> Convolution -> FFN -> LayerNorm
    """
    def __init__(self, dim, ffn_expansion_factor=4, conv_kernel_size=31, dropout_rate=0.1):
        super(MambaformerBlock, self).__init__()
        
        self.ffn1 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        
        # REPLACED: self.mhsa is now self.sequence_module
        self.sequence_module = MambaSequenceModule(dim, dropout_rate=dropout_rate)
        
        self.conv_module = ConvolutionModule(dim, conv_kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.sequence_module(x)
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class ChordFormer(nn.Module):
    """
    The updated model architecture using Mambaformer blocks.
    """
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 num_layers: int, 
                 output_dims: List[int], # Removed num_heads from args
                 ffn_expansion_factor: int = 4, 
                 conv_kernel_size: int = 31, 
                 dropout_rate: float = 0.1):
        super(ChordFormer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Swapped ConformerBlock for MambaformerBlock
        self.conformer_layers = nn.ModuleList([
            MambaformerBlock(
                dim=model_dim,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_kernel_size=conv_kernel_size,
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
    conv_kernel_size: int = 31,
    dropout_rate: float = 0.1
) -> ChordFormer:
    """
    Builds, initializes, and returns a ChordFormer model with the specified hyperparameters.
    
    Args:
        input_dim (int): Dimension of input features (CQT bins). Defaults to 252.
        model_dim (int): The internal working dimension of the model. Defaults to 256.
        num_layers (int): Number of Conformer blocks to stack. Defaults to 4.
        num_heads (int): Number of attention heads. Defaults to 4.
        output_dims (Optional[List[int]]): List of output dimensions for each chord component. 
                                            If None, uses default values from the paper.
        ffn_expansion_factor (int): Expansion factor for feed-forward layers. Defaults to 4.
        conv_kernel_size (int): Kernel size for the convolution module. Defaults to 31.
        dropout_rate (float): Dropout rate. Defaults to 0.1.

    Returns:
        ChordFormer: The initialized ChordFormer model.
    """
    # If no output dimensions are specified, use the defaults based on the paper's
    # structured chord representation, with the corrected logic for the first head.
    if output_dims is None:
        output_dims = [
            12 * 7 + 1,  # (12 roots * 7 triad qualities) + 1 for 'No Chord'
            12 + 1,      # 12 bass notes + N
            4,           # N, 7, b7, bb7
            4,           # N, 9, #9, b9
            3,           # N, 11, #11
            3            # N, 13, b13
        ]

    # Create the ChordFormer model instance
    model = ChordFormer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        output_dims=output_dims,
        ffn_expansion_factor=ffn_expansion_factor,
        conv_kernel_size=conv_kernel_size,
        dropout_rate=dropout_rate
    )
    
    # Initialize parameters with Xavier uniform distribution, similar to the example
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