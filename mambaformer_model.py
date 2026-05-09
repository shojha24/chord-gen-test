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


class PitchAwareEmbedding(nn.Module):
    """
    Replaces the standard linear input projection.
    Uses a 1D Convolution across the vertical pitch axis to explicitly 
    capture harmonic intervals (like octaves) before sequence modeling.
    """
    def __init__(self, input_dim=252, model_dim=256, bins_per_semitone=3):
        super(PitchAwareEmbedding, self).__init__()
        
        bins_per_octave = 12 * bins_per_semitone # 36
        
        # REMOVED padding="same" to avoid the CUDA memory crash
        self.pitch_filter = nn.Conv1d(
            in_channels=1, 
            out_channels=4, 
            kernel_size=bins_per_octave, 
            padding=0 
        )
        self.activation = nn.SiLU()
        self.projection = nn.Linear(4 * input_dim, model_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (Batch, Time, Pitch)
        B, T, P = x.shape
        
        x = x.view(B * T, 1, P)
        
        # EXPLICIT ASYMMETRIC PADDING: Left=17, Right=18 (Total 35)
        # Input length (252) + Padding (35) = 287. 
        # After Kernel (36): 287 - 36 + 1 = 252.
        x_padded = F.pad(x, (17, 18), mode="constant", value=0.0)
        
        x_filtered = self.pitch_filter(x_padded) 
        x_filtered = self.activation(x_filtered)
        
        x_flat = x_filtered.view(B, T, 4 * P)
        
        out = self.projection(x_flat)
        out = self.dropout(out)
        
        return out


class TransientHunter(nn.Module):
    """
    Applied EXACTLY ONCE globally. 
    A wide depthwise convolution to absorb sharp rhythmic attacks 
    and fast passing chords into the feature space before Mamba processes it.
    """
    def __init__(self, dim, kernel_size=31):
        super(TransientHunter, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        # Depthwise 1D Conv (groups=dim)
        self.acoustic_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # Conv1d expects (Batch, Channels, Time)
        x_transposed = x.transpose(1, 2)
        x_conv = self.acoustic_conv(x_transposed)
        
        # Transpose back to (Batch, Time, Channels)
        x_contextualized = x_conv.transpose(1, 2)
        x_contextualized = self.activation(x_contextualized)
        
        # Residual guarantees the sharp original frame isn't permanently smeared
        return residual + x_contextualized


class MambaSequenceModule(nn.Module):
    """
    The Bidirectional Mamba with the Dense Selective (DS) Gate merge.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout_rate=0.1):
        super(MambaSequenceModule, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
        # 1. Forward and Backward Mamba (Highly optimized defaults)
        self.mamba_forward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 2. The DS Gate Components (SIGMA Paper)
        self.ds_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.ds_linear = nn.Linear(dim, dim)
        self.silu = nn.SiLU()
        
        # 3. Final Projection
        self.mix_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # --- PATH A: Forward Mamba ---
        out_forward = self.mamba_forward(x)
        
        # --- PATH B: Backward Mamba ---
        # `.contiguous()` is critical here for memory alignment and speed!
        x_flipped = torch.flip(x, dims=[1]).contiguous()
        out_backward = self.mamba_backward(x_flipped)
        out_backward = torch.flip(out_backward, dims=[1]).contiguous()
        
        # --- PATH C: DS Gate Logic ---
        # The gate evaluates the local texture (kernel=3) to decide which direction to trust
        x_transposed = x.transpose(1, 2)
        gate_features = self.ds_conv(x_transposed).transpose(1, 2)
        
        gate_weights = torch.sigmoid(self.ds_linear(gate_features))
        gate_silu = self.silu(gate_features)
        
        # Dynamically blend forward and backward signals
        out = (gate_weights * out_forward) + (gate_silu * out_backward)
        
        out = self.mix_linear(out)
        out = self.dropout(out)
        
        return residual + out


class MambaformerBlock(nn.Module):
    """
    The Minimal Mambaformer Block: Bi-Mamba (with DS Gate) -> FFN -> LayerNorm
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, ffn_expansion_factor=4, dropout_rate=0.1):
        super(MambaformerBlock, self).__init__()
        
        # Pass the fat state parameters into the sequence module
        self.sequence_module = MambaSequenceModule(
            dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            dropout_rate=dropout_rate
        )
        self.ffn = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.sequence_module(x)
        x = self.ffn(x)
        x = self.final_layer_norm(x)
        return x


class ChordFormer(nn.Module):
    """
    The optimized architecture using the Transient Hunter and DS-Gated Mamba blocks.
    """
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 num_layers: int, 
                 output_dims: List[int],
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 ffn_expansion_factor: int = 4, 
                 dropout_rate: float = 0.1,
                 transient_kernel_size: int = 31): # NEW param
        super(ChordFormer, self).__init__()
        
        # 1. Frequency/Harmonic Extractor
        self.input_projection = PitchAwareEmbedding(input_dim=input_dim, model_dim=model_dim)
        
        # 2. Time/Transient Extractor (Applied EXACTLY ONCE)
        self.transient_hunter = TransientHunter(dim=model_dim, kernel_size=transient_kernel_size)

        # 3. Global Sequence Stack
        self.conformer_layers = nn.ModuleList([
            MambaformerBlock(
                dim=model_dim,
                d_state=d_state,  
                d_conv=d_conv,    
                expand=expand,    
                ffn_expansion_factor=ffn_expansion_factor,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        # 4. Decoding Heads
        self.output_heads = nn.ModuleList([
            nn.Linear(model_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transient_hunter(x)

        for layer in self.conformer_layers:
            x = layer(x)
        
        outputs = [head(x) for head in self.output_heads]
        return outputs


def build_chordformer(
    input_dim: int = 252,
    model_dim: int = 256,
    num_layers: int = 4,
    output_dims: Optional[List[int]] = None,
    d_state: int = 16,           
    d_conv: int = 4,
    expand: int = 2,             
    ffn_expansion_factor: int = 4,
    dropout_rate: float = 0.1,
    transient_kernel_size: int = 31
) -> ChordFormer:
    
    if output_dims is None:
        output_dims = [85, 13, 4, 4, 3, 3]

    model = ChordFormer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        output_dims=output_dims,
        d_state=d_state,         
        d_conv=d_conv,           
        expand=expand,           
        ffn_expansion_factor=ffn_expansion_factor,
        dropout_rate=dropout_rate,
        transient_kernel_size=transient_kernel_size
    )
    
    # Initialize standard layers
    for module in [model.input_projection, model.output_heads]:
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Initialize Mambaformer specific weights
    for layer in model.conformer_layers:
        for p in layer.ffn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize the new DS Gate Linear layers
        for p in layer.sequence_module.ds_linear.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in layer.sequence_module.mix_linear.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Leave layer.sequence_module.mamba_forward and mamba_backward alone
            
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