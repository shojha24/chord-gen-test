import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-computes cos/sin tables for Rotary Position Embedding.

    RoPE rotates pairs of dimensions by position-dependent angles.  Each
    pair (2i, 2i+1) gets its own frequency:

        θ_i = base^(-2i / head_dim)

    Returns cos and sin tensors of shape (seq_len, head_dim), where the
    second dimension already has the full rotation applied (pairs duplicated
    via [..., 0::2] and [..., 1::2] interleaving handled in apply_rope).
    """
    # Frequencies: one per pair of dimensions → (head_dim // 2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices: (seq_len,)
    positions = torch.arange(seq_len, device=device).float()

    # Outer product → (seq_len, head_dim // 2)
    freqs = torch.outer(positions, inv_freq)

    # Duplicate each frequency so the tensor matches full head_dim:
    # [f0, f1, f2, ...] → [f0, f0, f1, f1, f2, f2, ...]
    # This lets us apply the rotation without reshaping inside forward().
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)

    return emb.cos(), emb.sin()               # both (seq_len, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Given x of shape (..., head_dim), returns the tensor where each pair
    (x_{2i}, x_{2i+1}) is replaced by (-x_{2i+1}, x_{2i}).

    This is the 90-degree rotation component of RoPE.
    """
    # Split into first and second halves along the last axis
    x1 = x[..., : x.shape[-1] // 2]   # even-indexed dims
    x2 = x[..., x.shape[-1] // 2 :]   # odd-indexed dims
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to x.

    Args:
        x:   (B, T, dim)  — the tensor to rotate
        cos: (T, dim)     — cosine table from build_rope_cache
        sin: (T, dim)     — sine table from build_rope_cache

    Returns:
        Rotated tensor of the same shape as x.

    The rotation formula for position p is:
        x_rotated = x * cos(p) + rotate_half(x) * sin(p)
    """
    # Broadcast (T, dim) → (1, T, dim) so it works across the batch axis
    cos = cos.unsqueeze(0)   # (1, T, dim)
    sin = sin.unsqueeze(0)   # (1, T, dim)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Model modules  (unchanged from original except MambaSequenceModule)
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """
    Feed-Forward Network with pre-LayerNorm and optional Macaron half-step scaling.

    When used as a Macaron sandwich (one FFN before, one after the core module),
    both instances are constructed with scale=0.5.  The residual becomes:
        output = x + 0.5 * FFN(LayerNorm(x))
    which keeps the two FFN contributions equal and prevents either from
    dominating the gradient signal through the block.

    When used as a single full-step FFN (scale=1.0, the default), behaviour is
    identical to the original implementation.
    """
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.1, scale=1.0):
        super(FeedForwardModule, self).__init__()
        self.scale = scale
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
        return residual + self.scale * x


class PitchAwareEmbedding(nn.Module):
    """
    Replaces the standard linear input projection.
    Uses a 1D Convolution across the vertical pitch axis to explicitly 
    capture harmonic intervals (like octaves) before sequence modeling.
    """
    def __init__(self, input_dim=252, model_dim=256, bins_per_semitone=3):
        super(PitchAwareEmbedding, self).__init__()
        
        bins_per_octave = 12 * bins_per_semitone  # 36
        
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
        
        # Asymmetric padding: left=17, right=18 → output length stays P
        x_padded = F.pad(x, (17, 18), mode="constant", value=0.0)
        
        x_filtered = self.pitch_filter(x_padded)
        x_filtered = self.activation(x_filtered)
        
        x_flat = x_filtered.view(B, T, 4 * P)
        
        out = self.projection(x_flat)
        out = self.dropout(out)
        
        return out


class LocalContextConv(nn.Module):
    """
    Per-block local temporal convolution with GLU gating.

    Previously this was a plain depthwise conv applied once globally.  It now
    runs inside every MambaformerBlock, matching the Conformer's per-block
    ConvolutionModule topology, with the full pipeline restored:

        LayerNorm → pointwise expand (dim → 2*dim) → GLU → depthwise conv
                  → BatchNorm → SiLU → pointwise project (dim → dim) → residual

    The GLU is the critical addition: it multiplicatively gates the expanded
    features before the depthwise conv, letting each block decide which
    frequency-band / temporal patterns are relevant at that depth.  For chord
    estimation this matters because the useful spectral cues shift across
    layers — low-level transient suppression early, harmonic structure later.

    BatchNorm on the depthwise output (same as the Conformer) stabilises
    training at the cost of one small statistics buffer per block.
    """
    def __init__(self, dim, kernel_size=31):
        super(LocalContextConv, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)

        # Expand to 2*dim so GLU halves it back to dim
        self.pointwise_expand  = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu               = nn.GLU(dim=1)          # operates on channel axis

        self.depthwise_conv    = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim                                  # depthwise
        )
        self.batch_norm        = nn.BatchNorm1d(dim)
        self.activation        = nn.SiLU()

        self.pointwise_project = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout           = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        x = x.transpose(1, 2)              # (B, dim, T) for Conv1d
        x = self.pointwise_expand(x)       # (B, 2*dim, T)
        x = self.glu(x)                    # (B, dim, T)  — halves channels
        x = self.depthwise_conv(x)         # (B, dim, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_project(x)      # (B, dim, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)             # (B, T, dim)

        return residual + x


class MambaSequenceModule(nn.Module):
    """
    Bidirectional Mamba with the Dense Selective (DS) Gate merge.

    Change from original: RoPE is applied to g0 (the combined linear+conv
    signal) before it is passed to the gate linear.  This injects absolute
    positional information into the direction-selection decision — the gate
    can now choose forward vs. backward Mamba based on where in the sequence
    the frame sits, not just what the frame contains.

    RoPE is computed lazily and cached per (seq_len, device) to avoid
    rebuilding the table on every forward pass.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout_rate=0.1):
        super(MambaSequenceModule, self).__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim)
        
        self.mamba_forward  = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # DS Gate layers (paper order: linear → conv → gate)
        self.ds_dense       = nn.Linear(dim, dim)
        self.ds_conv        = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.ds_gate_linear = nn.Linear(dim, dim)   # W_δ: produces δ1

        self.mix_linear     = nn.Linear(dim, dim)
        self.dropout        = nn.Dropout(dropout_rate)

        # RoPE cache — populated lazily in forward()
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_seq_len: int = -1

    def _get_rope(self, seq_len: int, device: torch.device):
        """Returns cached RoPE tables, rebuilding only when seq_len changes."""
        if seq_len != self._rope_seq_len or self._rope_cos is None:
            self._rope_cos, self._rope_sin = build_rope_cache(seq_len, self.dim, device)
            self._rope_seq_len = seq_len
        return self._rope_cos, self._rope_sin

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        B, T, _ = x.shape

        # --- Bi-directional Mamba ---
        out_forward = self.mamba_forward(x)

        x_flipped   = torch.flip(x, dims=[1]).contiguous()
        out_backward = torch.flip(
            self.mamba_backward(x_flipped), dims=[1]
        ).contiguous()

        # --- DS Gate (with RoPE) ---
        # Step 1: linear + conv to get the base gate signal
        g0 = self.ds_conv(
            self.ds_dense(x).transpose(1, 2)
        ).transpose(1, 2)                              # (B, T, dim)

        # Step 2: apply RoPE to g0 before the gate linear
        #   → each frame's gate decision is now modulated by its position
        cos, sin = self._get_rope(T, x.device)
        g0_rope = apply_rope(g0, cos, sin)             # (B, T, dim)

        # Step 3: gate linear on the position-aware signal
        delta1 = self.ds_gate_linear(g0_rope)          # (B, T, dim)

        # Soft direction switch: g → forward weight, (1-g) → backward weight
        g   = torch.sigmoid(delta1)
        out = g * out_forward + (1 - g) * out_backward

        out = self.mix_linear(out)
        out = self.dropout(out)
        
        return residual + out


class MambaformerBlock(nn.Module):
    """
    Full Macaron-style Mambaformer block:

        FFN (×0.5) → Bi-Mamba (DS Gate + RoPE) → LocalContextConv → FFN (×0.5) → LayerNorm

    This matches the Conformer block topology exactly, replacing only MHSA
    with Bi-Mamba.  The two ×0.5 FFNs sandwich the core sequence module,
    balancing gradient flow between the feed-forward and sequence modeling
    pathways.  LocalContextConv (now with GLU) runs per-block so every layer
    gets its own local temporal refinement, not just the first.

    d_state is accepted per-block to support the layer-wise schedule (#5):
    early blocks use small d_state for local transients; later blocks use
    large d_state for long-range harmonic context.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2,
                 ffn_expansion_factor=4, conv_kernel_size=31, dropout_rate=0.1):
        super(MambaformerBlock, self).__init__()

        self.ffn1 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate, scale=0.5)
        self.sequence_module = MambaSequenceModule(
            dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout_rate=dropout_rate
        )
        self.conv_module = LocalContextConv(dim, kernel_size=conv_kernel_size)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate, scale=0.5)
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
    ChordFormer: PitchAwareEmbedding → N × MambaformerBlock → multi-head output.

    Changes from the RoPE-only version:
    - Global transient_hunter removed.  LocalContextConv now lives inside each
      MambaformerBlock, so every layer performs its own local temporal refinement.
    - d_state_schedule: a per-layer list of SSM state sizes.  Early layers use
      small states (fast, local); later layers use large states (slow, global).
      Passing a single int falls back to uniform d_state across all blocks.
    - conv_kernel_size forwarded into each block's LocalContextConv.
    """
    def __init__(self,
                 input_dim: int,
                 model_dim: int,
                 num_layers: int,
                 output_dims: List[int],
                 d_state_schedule: List[int],           # one entry per layer
                 d_conv: int = 4,
                 expand: int = 2,
                 ffn_expansion_factor: int = 4,
                 conv_kernel_size: int = 31,
                 dropout_rate: float = 0.1):
        super(ChordFormer, self).__init__()

        assert len(d_state_schedule) == num_layers, (
            f"d_state_schedule must have exactly num_layers={num_layers} entries, "
            f"got {len(d_state_schedule)}"
        )

        self.input_projection = PitchAwareEmbedding(input_dim=input_dim, model_dim=model_dim)
        # No global transient_hunter — conv is per-block now

        self.conformer_layers = nn.ModuleList([
            MambaformerBlock(
                dim=model_dim,
                d_state=d_state_schedule[i],            # layer-specific state size
                d_conv=d_conv,
                expand=expand,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate
            )
            for i in range(num_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(model_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.conformer_layers:
            x = layer(x)
        return [head(x) for head in self.output_heads]


def build_chordformer(
    input_dim: int = 252,
    model_dim: int = 256,
    num_layers: int = 4,
    output_dims: Optional[List[int]] = None,
    d_state_schedule: Optional[List[int]] = None,   # per-layer SSM state sizes
    d_conv: int = 4,
    expand: int = 2,
    ffn_expansion_factor: int = 4,
    conv_kernel_size: int = 31,
    dropout_rate: float = 0.1,
) -> ChordFormer:
    """
    Builds and initialises a ChordFormer.

    d_state_schedule controls per-layer SSM state sizes (#5).  The default
    [16, 16, 64, 64] gives early layers small states for local transient
    detection and late layers large states for long-range harmonic context.
    Pass a list of length num_layers to override.  A single repeated value
    (e.g. [16]*4) reproduces the original uniform behaviour.
    """
    if output_dims is None:
        output_dims = [85, 13, 4, 4, 3, 3]

    if d_state_schedule is None:
        # Default: small → large, doubling at the halfway point
        half = num_layers // 2
        d_state_schedule = [16] * half + [64] * (num_layers - half)

    assert len(d_state_schedule) == num_layers, (
        f"d_state_schedule length ({len(d_state_schedule)}) must equal "
        f"num_layers ({num_layers})"
    )

    model = ChordFormer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        output_dims=output_dims,
        d_state_schedule=d_state_schedule,
        d_conv=d_conv,
        expand=expand,
        ffn_expansion_factor=ffn_expansion_factor,
        conv_kernel_size=conv_kernel_size,
        dropout_rate=dropout_rate,
    )

    # --- Initialisation ---
    # Input projection and output heads
    for module in [model.input_projection, model.output_heads]:
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    for layer in model.conformer_layers:
        # Both Macaron FFNs
        for ffn in [layer.ffn1, layer.ffn2]:
            for p in ffn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # DS Gate layers
        for p in layer.sequence_module.ds_dense.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in layer.sequence_module.ds_gate_linear.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if p.dim() == 1:
                # Bias = 0 → sigmoid(0) = 0.5: balanced forward/backward blend at init
                nn.init.constant_(p, 0.0)
        for p in layer.sequence_module.mix_linear.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # LocalContextConv pointwise layers (depthwise and BN left at default init)
        for p in layer.conv_module.pointwise_expand.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in layer.conv_module.pointwise_project.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_chordformer()
    print(model)

    dummy_cqt = torch.randn(8, 1000, 252)
    preds = model(dummy_cqt)

    print("\n--- Test Run ---")
    print(f"Input shape: {dummy_cqt.shape}")
    print("Output shapes for each chord component head:")
    for i, p in enumerate(preds):
        print(f"  Head {i+1}: {p.shape}")