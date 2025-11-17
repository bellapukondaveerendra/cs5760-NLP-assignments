import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Implement scaled dot-product attention
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)
    
    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention weight matrix
    """
    
    # Get dimension
    d_k = Q.size(-1)
    
    # Step 1: Compute Q * K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k)
    scaled_scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # Step 4: Multiply by V
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    
    # Create test inputs
    batch_size = 2
    seq_len = 5
    d_k = 8
    d_v = 8
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print("Input shapes:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    
    # Run attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"\nAttention weight matrix (first sample):")
    print(attention_weights[0])
    print(f"\nOutput vectors (first sample):")
    print(output[0])
    
    # Softmax stability check
    print("\n" + "="*60)
    print("SOFTMAX STABILITY CHECK")
    print("="*60)
    
    # Before scaling
    unscaled_scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"\nBefore scaling (QK^T):")
    print(f"  Score range: [{unscaled_scores.min().item():.4f}, {unscaled_scores.max().item():.4f}]")
    unscaled_softmax = F.softmax(unscaled_scores, dim=-1)
    print(f"  Max softmax value: {unscaled_softmax.max().item():.6f}")
    
    # After scaling
    scaled_scores = unscaled_scores / math.sqrt(d_k)
    print(f"\nAfter scaling (QK^T / sqrt({d_k})):")
    print(f"  Score range: [{scaled_scores.min().item():.4f}, {scaled_scores.max().item():.4f}]")
    scaled_softmax = F.softmax(scaled_scores, dim=-1)
    print(f"  Max softmax value: {scaled_softmax.max().item():.6f}")
    
    print(f"\nScaling prevents softmax saturation and maintains gradient flow.")