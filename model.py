import torch
import torch.nn as nn
from typing import List, Optional


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


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Headed Self-Attention module with pre-LayerNorm.
    """
    def __init__(self, dim, num_heads=4, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.layer_norm = nn.LayerNorm(dim)
        # A single linear layer is more efficient for generating Q, K, V
        self.qkv_linear = nn.Linear(dim, dim * 3, bias=False)
        self.out_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        batch_size, seq_length, _ = x.shape
        
        # Create Q, K, V from a single linear projection
        qkv = self.qkv_linear(x).chunk(3, dim=-1)
        # Reshape and transpose for multi-head attention
        Q, K, V = [t.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim)
        output = self.dropout(self.out_linear(attn_output))
        
        return residual + output


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


class ConformerBlock(nn.Module):
    """
    The complete Conformer block, sequencing the modules as per the paper:
    FFN -> Attention -> Convolution -> FFN -> LayerNorm
    """
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=4, conv_kernel_size=31, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()
        
        self.ffn1 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout_rate)
        self.conv_module = ConvolutionModule(dim, conv_kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion_factor, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class ChordFormer(nn.Module):
    """
    The complete ChordFormer model architecture.
    It includes an input projection layer, a stack of Conformer blocks,
    and a multi-headed output layer for structured chord recognition.
    """
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 num_layers: int, 
                 num_heads: int, 
                 output_dims: List[int],
                 ffn_expansion_factor: int = 4, 
                 conv_kernel_size: int = 31, 
                 dropout_rate: float = 0.1):
        """
        Args:
            input_dim (int): Dimension of the input features (e.g., CQT bins).
            model_dim (int): The internal dimension of the model (d_model).
            num_layers (int): The number of Conformer blocks to stack.
            num_heads (int): The number of attention heads.
            output_dims (List[int]): A list of output dimensions for each chord 
                                     component head. For ChordFormer, this would be 
                                     a list of 6 integers.
            ffn_expansion_factor (int): Expansion factor for the FFNs.
            conv_kernel_size (int): Kernel size for the convolution module.
            dropout_rate (float): Dropout rate.
        """
        super(ChordFormer, self).__init__()
        
        # 1. Input Projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 2. Conformer Encoder Stack
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=model_dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        # 3. Decoding Heads
        # Create a separate linear layer for each of the 6 chord components
        self.output_heads = nn.ModuleList([
            nn.Linear(model_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        # x shape: (batch_size, sequence_length, model_dim)

        # Pass through the stack of Conformer blocks
        for layer in self.conformer_layers:
            x = layer(x)
        
        # Pass the final output through each decoding head
        # This will produce a list of tensors, one for each chord component
        outputs = [head(x) for head in self.output_heads]
        
        return outputs


def build_chordformer(
    input_dim: int = 252,
    model_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
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
        num_heads=num_heads,
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


# --- Example Usage ---
# Build the model using the default hyperparameters from the paper [attached_file:1]
chordformer_model = build_chordformer()

# Print the model to verify its structure
print(chordformer_model)

# Test with a dummy input tensor
dummy_cqt = torch.randn(8, 1000, 252) # (batch, sequence_length, cqt_bins)
predictions = chordformer_model(dummy_cqt)

print("\n--- Test Run ---")
print(f"Input shape: {dummy_cqt.shape}")
print("Output shapes for each chord component head:")
for i, p in enumerate(predictions):
    print(f"  Head {i+1}: {p.shape}")