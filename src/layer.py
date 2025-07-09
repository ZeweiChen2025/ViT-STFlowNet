import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .cell_utils import get_2d_sin_cos_pos_embed

__all__ = ['Decoder', 'Encoder']


class Attention(nn.Module):
    """Multi-head self-attention module with layer normalization and dropout"""
    
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 dropout_rate=1.0,
                 compute_dtype=torch.float16):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        # Maintain FP32 version for numerical stability
        self.embed_dim_per_head_fp32 = torch.tensor(self.embed_dim_per_head, dtype=torch.float32)
        self.mlp_ratio = mlp_ratio
        self.compute_dtype = compute_dtype

        # Layer normalization for input stabilization
        self.layer_norm = nn.LayerNorm([embed_dim], eps=1e-6)

        # Query/Key/Value projection layers
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.query.weight)  # Xavier initialization

        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.key.weight)

        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.value.weight)

        # Output projection
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.proj.weight)

        # Regularization
        self.attention_dropout_rate = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        # Softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """
        Reshape and transpose input for multi-head attention computation
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Reshaped tensor of shape [batch_size, num_heads, seq_len, embed_dim_per_head]
        """
        # Get input dimensions excluding last dimension
        x_shape = x.shape
        # Reshape to separate head dimension
        new_x_shape = x_shape[:-1] + (self.num_heads, self.embed_dim_per_head)

        # Reshape and transpose dimensions
        x = torch.reshape(x, new_x_shape)
        # Transpose to [batch_size, num_heads, seq_len, embed_dim_per_head]
        x = torch.transpose(x, 1, 2)

        return x

    def forward(self, x):
        """Forward pass through attention mechanism"""
        # Project inputs to query/key/value
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        # Reshape for multi-head computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        # Scale attention scores for stability
        scaled_attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))

        # Compute attention probabilities
        attention_probs = self.softmax(scaled_attention_scores)
        attention_probs = self.attention_dropout_rate(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Ensure memory layout
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final projection and dropout
        attention_output = self.proj(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class Mlp(nn.Module):
    """Feed-forward network (MLP) with GELU activation and dropout"""
    
    def __init__(self, embed_dim, mlp_ratio, dropout_rate=1.0, compute_dtype=torch.float16):
        super(Mlp, self).__init__()
        # First fully-connected layer with expansion
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier initialization

        # Second fully-connected layer (projection back to embed_dim)
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Activation and regularization
        self.act_fn = nn.GELU()  # Gaussian Error Linear Unit
        self.dropout = nn.Dropout(dropout_rate)

        # Computation precision
        self.compute_dtype = compute_dtype

    def forward(self, x):
        """Forward pass through MLP"""
        # First FC layer with activation and dropout
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        
        # Second FC layer with dropout
        x = self.fc2(x)
        x = self.dropout(x)

        # Convert to specified computation dtype
        x = x.to(self.compute_dtype)

        return x


class Block(nn.Module):
    """Transformer block combining attention and feed-forward network with residual connections"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_rate=1.0, compute_dtype=torch.float16):
        super(Block, self).__init__()
        self.embed_dim = embed_dim

        # Pre-attention layer normalization
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Pre-MLP layer normalization
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Feed-forward network
        self.ffn = Mlp(embed_dim, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)

        # Attention mechanism
        self.attn = Attention(embed_dim, num_heads, dropout_rate, compute_dtype=compute_dtype)

    def forward(self, x):
        """Forward pass through transformer block"""
        # Attention sub-layer with residual connection
        h = x  # Residual
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h  # Add residual

        # Feed-forward sub-layer with residual connection
        h = x  # Residual
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # Add residual

        return x


class Embedding(nn.Module):
    """Patch embedding layer with learned position embeddings"""
    
    def __init__(self, input_dims, embed_dim, patch_size=(16, 16), compute_dtype=torch.float16):
        super(Embedding, self).__init__()
        self.compute_dtype = compute_dtype
        
        # Convolutional patch embedding (equivalent to linear projection of flattened patches)
        self.patch_embedding = nn.Conv2d(
            in_channels=input_dims,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,  # No padding since stride equals kernel size
            bias=True)  # Include bias term
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)  # Zero-initialize bias

    def forward(self, x):
        """Forward pass through patch embedding"""
        # Extract patches via convolution
        x = self.patch_embedding(x)
        
        # Flatten spatial dimensions (height, width) into sequence length
        x = x.view(x.size(0), x.size(1), -1)
        
        # Transpose to [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1)
        
        return x


class Encoder(nn.Module):
    """Transformer encoder with patch embedding and positional encoding"""
    
    def __init__(self, grid_size, in_channels, patch_size, depths, embed_dim, num_heads, mlp_ratio=4, dropout_rate=1.0,
                 compute_dtype=torch.float16):
        super(Encoder, self).__init__()
        # Patch embedding layer
        self.patch_embedding = Embedding(in_channels, embed_dim, patch_size, compute_dtype=compute_dtype)
        
        # Architecture parameters
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype

        # Fixed 2D sinusoidal positional embeddings
        pos_embed = get_2d_sin_cos_pos_embed(self.embed_dim, grid_size)
        self.position_embedding = nn.Parameter(torch.tensor(pos_embed, dtype=torch.float32), requires_grad=False)

        # Stack of transformer blocks
        self.layer = nn.ModuleList(
            [Block(self.embed_dim, num_heads, self.mlp_ratio, self.dropout_rate, compute_dtype) 
             for _ in range(self.depths)])

        # Final layer normalization
        self.encoder_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x):
        """Forward pass through encoder"""
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add positional embeddings
        x = x + self.position_embedding
        
        # Process through transformer blocks
        for layer_block in self.layer:
            x = layer_block(x)
            
        # Final normalization
        x = self.encoder_norm(x)
        
        return x


class Decoder(nn.Module):
    """Transformer decoder with positional encoding"""
    
    def __init__(self, grid_size, depths, embed_dim, num_heads, mlp_ratio=4, dropout_rate=1.0,
                 compute_dtype=torch.float16):
        super(Decoder, self).__init__()
        # Architecture parameters
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype

        # Fixed 2D sinusoidal positional embeddings
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.position_embedding = nn.Parameter(torch.tensor(pos_embed, dtype=torch.float32), requires_grad=False)

        # Stack of transformer blocks
        self.layer = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout_rate, compute_dtype) 
             for _ in range(depths)])

        # Final layer normalization
        self.decoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        """Forward pass through decoder"""
        # Add positional embeddings
        x = x + self.position_embedding
        
        # Process through transformer blocks
        for layer_block in self.layer:
            x = layer_block(x)
            
        # Final normalization
        x = self.decoder_norm(x)
        
        return x

