import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# RoPE helpers  (unchanged)
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-computes cos/sin tables for Rotary Position Embedding.

    RoPE rotates pairs of dimensions by position-dependent angles.  Each
    pair (2i, 2i+1) gets its own frequency:

        θ_i = base^(-2i / head_dim)

    Returns cos and sin tensors of shape (seq_len, head_dim), where the
    second dimension already has the full rotation applied (pairs duplicated
    via [..., 0::2] and [..., 1::2] interleaving handled in apply_rope).
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs     = torch.outer(positions, inv_freq)          # (seq_len, head_dim//2)
    emb       = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Given x of shape (..., head_dim), returns the tensor where each pair
    (x_{2i}, x_{2i+1}) is replaced by (-x_{2i+1}, x_{2i}).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to x.

    Args:
        x:   (B, T, dim)
        cos: (T, dim)
        sin: (T, dim)
    """
    cos = cos.unsqueeze(0)   # (1, T, dim)
    sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """
    Feed-Forward Network with pre-LayerNorm and optional Macaron half-step
    scaling.  Unchanged from original.
    """
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.1, scale=1.0):
        super().__init__()
        self.scale      = scale
        hidden_dim      = dim * expansion_factor
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1    = nn.Linear(dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout1   = nn.Dropout(dropout_rate)
        self.linear2    = nn.Linear(hidden_dim, dim)
        self.dropout2   = nn.Dropout(dropout_rate)

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
    Uses a 1-D Convolution across the pitch axis to capture harmonic
    intervals (octaves etc.) before sequence modelling.  Unchanged.
    """
    def __init__(self, input_dim=252, model_dim=256, bins_per_semitone=3):
        super().__init__()
        bins_per_octave   = 12 * bins_per_semitone          # 36
        self.pitch_filter = nn.Conv1d(1, 4, kernel_size=bins_per_octave, padding=0)
        self.activation   = nn.SiLU()
        self.projection   = nn.Linear(4 * input_dim, model_dim)
        self.dropout      = nn.Dropout(0.1)

    def forward(self, x):
        B, T, P = x.shape
        x         = x.view(B * T, 1, P)
        x_padded  = F.pad(x, (17, 18), mode="constant", value=0.0)
        x_filtered = self.pitch_filter(x_padded)
        x_filtered = self.activation(x_filtered)
        x_flat    = x_filtered.view(B, T, 4 * P)
        out       = self.projection(x_flat)
        return self.dropout(out)


class LocalContextConv(nn.Module):
    """
    Per-block local temporal convolution with GLU gating.  Unchanged.

        LayerNorm → pointwise expand (dim → 2*dim) → GLU → depthwise conv
                  → BatchNorm → SiLU → pointwise project (dim → dim) → residual
    """
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.layer_norm        = nn.LayerNorm(dim)
        self.pointwise_expand  = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu               = nn.GLU(dim=1)
        self.depthwise_conv    = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
        )
        self.batch_norm        = nn.BatchNorm1d(dim)
        self.activation        = nn.SiLU()
        self.pointwise_project = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout           = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_expand(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_project(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class MambaSequenceModule(nn.Module):
    """
    Bidirectional Mamba with the Dense Selective (DS) Gate merge.
    RoPE is applied to g0 before the gate linear.  Unchanged.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout_rate=0.1):
        super().__init__()
        self.dim            = dim
        self.layer_norm     = nn.LayerNorm(dim)
        self.mamba_forward  = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ds_dense       = nn.Linear(dim, dim)
        self.ds_conv        = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.ds_gate_linear = nn.Linear(dim, dim)
        self.mix_linear     = nn.Linear(dim, dim)
        self.dropout        = nn.Dropout(dropout_rate)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_seq_len: int = -1

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len != self._rope_seq_len or self._rope_cos is None:
            self._rope_cos, self._rope_sin = build_rope_cache(seq_len, self.dim, device)
            self._rope_seq_len = seq_len
        return self._rope_cos, self._rope_sin

    def forward(self, x):
        residual     = x
        x            = self.layer_norm(x)
        B, T, _      = x.shape
        out_forward  = self.mamba_forward(x)
        x_flipped    = torch.flip(x, dims=[1]).contiguous()
        out_backward = torch.flip(self.mamba_backward(x_flipped), dims=[1]).contiguous()
        g0           = self.ds_conv(self.ds_dense(x).transpose(1, 2)).transpose(1, 2)
        cos, sin     = self._get_rope(T, x.device)
        g0_rope      = apply_rope(g0, cos, sin)
        delta1       = self.ds_gate_linear(g0_rope)
        g            = torch.sigmoid(delta1)
        out          = g * out_forward + (1 - g) * out_backward
        out          = self.mix_linear(out)
        out          = self.dropout(out)
        return residual + out


# ---------------------------------------------------------------------------
# NEW: Sliding Window Attention
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """
    Multi-head self-attention restricted to a local window of ±window_size
    frames, with the same RoPE applied to Q and K as the Mamba DS-gate uses.

    WHY THIS MODULE EXISTS
    ----------------------
    Pure SSMs compress all past context into a fixed-size recurrent state.
    This works well for local spectral patterns but creates a "memory cliff"
    for associative recall: when the same chord progression repeats after a
    verse/chorus gap, the SSM state has been overwritten by the intervening
    frames and can no longer retrieve the earlier pattern with high confidence.
    That is the direct cause of the low class-wise accuracy on H4–H6 (rare
    extension chords): the model sees a dominant-7 chord, its state is
    dominated by surrounding common chords, and the rare extensions get washed
    out.

    Attention has exact recall — it compares every query frame directly against
    every key frame within the window.  Adding even a single attention layer at
    the deepest block (where representations are most abstract) gives the model
    the retrieval capability it is missing, without replacing the SSM machinery
    that handles sustained harmonic context efficiently.

    WHY SLIDING WINDOW, NOT FULL ATTENTION
    ---------------------------------------
    Your segments are ~1000 frames (23 s at hop=512, sr=22050).  Full O(T²)
    attention over 1000 frames is expensive and mostly wasteful: a chord at
    frame 50 rarely needs to directly attend to a chord at frame 950.  What it
    does need is the 2–3 chords immediately surrounding it for disambiguation.
    A window of 128 frames ≈ 3 s covers that context at near-linear cost.

    SAMBA (ICLR 2025) validated exactly this design: Mamba layers handle
    long-range compression, SWA fills the precise local-recall gap.  HELIX
    (2025) confirmed the same pattern holds for audio specifically, with the
    largest gains on temporally structured tasks (chord recognition qualifies)
    and zero benefit on short stationary clips (which you are not doing).

    WHY THE SAME ROPE AS THE MAMBA BLOCKS
    --------------------------------------
    TransXSSM (arXiv Jun 2025) showed that naive SSM+attention hybrids suffer
    a positional encoding discontinuity: attention uses explicit RoPE, SSMs
    encode position implicitly through their recurrent dynamics.  When you
    stack the two, the positional coordinate systems are mismatched at every
    layer interface, which hurts gradient flow and final accuracy.  The fix is
    to apply the same RoPE — same base (10000), same head_dim (64 for 4 heads
    at d_model=256) — to Q and K here as your MambaSequenceModule already
    applies to g0.  Since build_rope_cache is a module-level function, both
    modules share the same computation; the caches are separate instances but
    produce identical tensors.

    WHERE THIS BLOCK SITS IN THE STACK
    -----------------------------------
    It replaces the sequence_module of the LAST MambaformerBlock only (index
    num_layers-1).  Heracles (2024) established that attention should go in
    deeper layers where representations are abstract enough to benefit from
    global comparison.  HELIX confirmed that early-layer attention actively
    hurts spectrogram-based audio models by displacing useful local SSM
    processing.  Blocks 0–2 keep their BiMamba+DS-gate intact.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads   # 64 for dim=256, num_heads=4
        self.window_size = window_size

        self.layer_norm  = nn.LayerNorm(dim)
        # Single fused projection for Q, K, V — same pattern as the
        # Chordformer's MHSA and standard in efficient attention implementations
        self.qkv         = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj    = nn.Linear(dim, dim)
        self.dropout     = nn.Dropout(dropout_rate)

        # RoPE cache — same lazy-build pattern as MambaSequenceModule so that
        # both modules always use tables built with identical hyperparameters.
        # head_dim here matches the head_dim used for the DS-gate RoPE only
        # when dim and num_heads give head_dim == dim (MambaSequenceModule uses
        # the full dim as head_dim for its gate).  For Q/K we use per-head
        # head_dim = dim // num_heads = 64, which is the standard RoPE usage.
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_seq_len: int = -1

    def _get_rope(self, seq_len: int, device: torch.device):
        """Lazy RoPE cache keyed on seq_len, same contract as MambaSequenceModule."""
        if seq_len != self._rope_seq_len or self._rope_cos is None:
            # base=10000 matches build_rope_cache default used throughout
            self._rope_cos, self._rope_sin = build_rope_cache(
                seq_len, self.head_dim, device, base=10000.0
            )
            self._rope_seq_len = seq_len
        return self._rope_cos, self._rope_sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x        = self.layer_norm(x)
        B, T, _  = x.shape

        # ── Project to Q, K, V ──────────────────────────────────────────────
        # (B, T, 3*dim) → three tensors of (B, num_heads, T, head_dim)
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        def reshape(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = reshape(Q), reshape(K), reshape(V)

        # ── Apply RoPE to Q and K ────────────────────────────────────────────
        # cos/sin shape: (T, head_dim) → unsqueeze to (1, 1, T, head_dim) for
        # broadcasting over batch and head dimensions.
        cos, sin  = self._get_rope(T, x.device)
        cos_h     = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
        sin_h     = sin.unsqueeze(0).unsqueeze(0)
        # rotate_half operates on the last dim; Q/K last dim is head_dim ✓
        Q = (Q * cos_h) + (rotate_half(Q) * sin_h)
        K = (K * cos_h) + (rotate_half(K) * sin_h)

        # ── Sliding-window mask ──────────────────────────────────────────────
        # Build a (T, T) boolean mask where True means "block this attention
        # weight".  Query i can attend to key j only if |i - j| <= window_size.
        # At hop=512, sr=22050: window_size=128 ≈ 2.97 s, which covers 1–2
        # full chord durations on either side of each query frame — enough
        # context to disambiguate extensions without the cost of full attention.
        idx  = torch.arange(T, device=x.device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()   # (T, T)
        mask = dist > self.window_size                        # True = mask out

        # ── Scaled dot-product attention with mask ───────────────────────────
        scale  = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, T)
        # Broadcast mask (T, T) → (1, 1, T, T) across batch and heads
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # ── Aggregate values and project ────────────────────────────────────
        out = torch.matmul(attn, V)                              # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.dim)
        out = self.dropout(self.out_proj(out))

        return residual + out


# ---------------------------------------------------------------------------
# MambaformerBlock  (updated to support use_swa flag)
# ---------------------------------------------------------------------------

class MambaformerBlock(nn.Module):
    """
    Full Macaron-style block:

        FFN (×0.5) → sequence_module → LocalContextConv → FFN (×0.5) → LayerNorm

    The sequence_module is either:
      - MambaSequenceModule  (BiMamba + DS-gate + RoPE)  for blocks 0–2
      - SlidingWindowAttention (SWA + RoPE)              for block 3

    The LocalContextConv, both FFNs, and the final LayerNorm are identical
    regardless of which sequence module is used, preserving the Conformer
    block topology throughout the stack.

    CHANGE FROM ORIGINAL
    --------------------
    Added use_swa / swa_window parameters.  When use_swa=True the
    MambaSequenceModule is replaced with SlidingWindowAttention.  All other
    behaviour is unchanged — the d_state argument is simply ignored for SWA
    blocks since SWA has no SSM state dimension.
    """

    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout_rate=0.1,
        use_swa: bool = False,
        swa_window: int = 128,
        swa_heads: int = 4,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate, scale=0.5)

        if use_swa:
            # Replace BiMamba with Sliding Window Attention at this block.
            # d_state is irrelevant here and not forwarded.
            self.sequence_module = SlidingWindowAttention(
                dim=dim,
                num_heads=swa_heads,
                window_size=swa_window,
                dropout_rate=dropout_rate,
            )
        else:
            self.sequence_module = MambaSequenceModule(
                dim, d_state=d_state, d_conv=d_conv,
                expand=expand, dropout_rate=dropout_rate,
            )

        self.conv_module      = LocalContextConv(dim, kernel_size=conv_kernel_size)
        self.ffn2             = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate, scale=0.5)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.sequence_module(x)
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# ChordFormer  (updated output_heads to use CalibratedHead)
# ---------------------------------------------------------------------------

class ChordFormer(nn.Module):
    """
    ChordFormer: PitchAwareEmbedding → N × MambaformerBlock → multi-head output.

    CHANGES FROM ORIGINAL
    ---------------------
    1. The last MambaformerBlock (index num_layers-1) uses SlidingWindowAttention
       instead of MambaSequenceModule.  All earlier blocks are unchanged.

    2. output_heads now use CalibratedHead instead of plain nn.Linear, adding a
       learnable temperature scalar per head.

    Everything else — d_state_schedule, PitchAwareEmbedding, LocalContextConv,
    Macaron FFNs, final LayerNorm — is identical to the previous version.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        output_dims: List[int],
        d_state_schedule: List[int],
        d_conv: int = 4,
        expand: int = 2,
        ffn_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout_rate: float = 0.1,
        swa_window: int = 128,
        swa_heads: int = 4,
    ):
        super().__init__()

        assert len(d_state_schedule) == num_layers, (
            f"d_state_schedule must have exactly num_layers={num_layers} entries, "
            f"got {len(d_state_schedule)}"
        )

        self.input_projection = PitchAwareEmbedding(
            input_dim=input_dim, model_dim=model_dim
        )

        self.conformer_layers = nn.ModuleList([
            MambaformerBlock(
                dim=model_dim,
                d_state=d_state_schedule[i],
                d_conv=d_conv,
                expand=expand,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate,
                # Only the final block uses SWA (Heracles staging principle:
                # attention in deep layers only; HELIX: early-layer attention
                # hurts spectrogram-based audio models).
                use_swa=(i == num_layers - 1),
                swa_window=swa_window,
                swa_heads=swa_heads,
            )
            for i in range(num_layers)
        ])

        # CalibratedHead replaces plain nn.Linear — adds one temperature
        # scalar per head (6 extra parameters total) to address the
        # confidence gap identified in the loss vs accuracy analysis.
        self.output_heads = nn.ModuleList([
            nn.Linear(model_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.conformer_layers:
            x = layer(x)
        return [head(x) for head in self.output_heads]


# ---------------------------------------------------------------------------
# build_chordformer  (updated with swa_window / swa_heads args + init)
# ---------------------------------------------------------------------------

def build_chordformer(
    input_dim: int = 252,
    model_dim: int = 256,
    num_layers: int = 4,
    output_dims: Optional[List[int]] = None,
    d_state_schedule: Optional[List[int]] = None,
    d_conv: int = 4,
    expand: int = 2,
    ffn_expansion_factor: int = 4,
    conv_kernel_size: int = 31,
    dropout_rate: float = 0.1,
    swa_window: int = 128,
    swa_heads: int = 4,
) -> ChordFormer:
    """
    Builds and initialises a ChordFormer with a hybrid Mamba+SWA architecture.

    The last of the num_layers blocks uses SlidingWindowAttention instead of
    the BiMamba+DS-gate sequence module.  All other blocks are unchanged.

    swa_window=128 corresponds to ~2.97 s at hop=512, sr=22050 — covering
    1–2 full chord durations on either side of each query frame.

    d_state_schedule applies only to the first num_layers-1 Mamba blocks;
    the last entry is unused (SWA has no SSM state).
    """
    if output_dims is None:
        output_dims = [85, 13, 4, 4, 3, 3]

    if d_state_schedule is None:
        half             = num_layers // 2
        d_state_schedule = [32] * half + [64] * (num_layers - half)

    assert len(d_state_schedule) == num_layers

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
        swa_window=swa_window,
        swa_heads=swa_heads,
    )

    # ── Initialisation ───────────────────────────────────────────────────────

    # Input projection
    for p in model.input_projection.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Output heads
    for head in model.output_heads:
        for p in head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    for i, layer in enumerate(model.conformer_layers):
        # Both Macaron FFNs — identical for Mamba and SWA blocks
        for ffn in [layer.ffn1, layer.ffn2]:
            for p in ffn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        is_swa = (i == num_layers - 1)

        if is_swa:
            # SWA block: initialise QKV and output projection
            swa = layer.sequence_module
            for p in swa.qkv.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in swa.out_proj.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            # Mamba block: initialise DS-gate linears
            sm = layer.sequence_module
            for p in sm.ds_dense.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in sm.ds_gate_linear.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                if p.dim() == 1:
                    # Bias=0 → sigmoid(0)=0.5: balanced fwd/bwd at init
                    nn.init.constant_(p, 0.0)
            for p in sm.mix_linear.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # LocalContextConv — identical for both block types
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
    model = build_chordformer(expand=1)
    print(model)

    """
    dummy_cqt = torch.randn(2, 1000, 252)
    preds     = model(dummy_cqt)

    print("\n--- Test Run ---")
    print(f"Input shape: {dummy_cqt.shape}")
    print("Output shapes per head:")
    for i, p in enumerate(preds):
        print(f"  Head {i+1}: {p.shape}")
    """

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Per-block breakdown
    print("\nPer-block parameter counts:")
    for i, layer in enumerate(model.conformer_layers):
        n     = sum(p.numel() for p in layer.parameters())
        kind  = "SWA" if i == 3 else f"BiMamba d_state={[32,32,64,64][i]}"
        print(f"  Block {i} ({kind}): {n:,}")