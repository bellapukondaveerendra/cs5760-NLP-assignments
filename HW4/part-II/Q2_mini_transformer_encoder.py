import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.W_o(x)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with Add & Norm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with Add & Norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, 
                 num_layers=2, d_ff=512, max_len=100, dropout=0.1):
        super(MiniTransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        all_attention_weights = []
        
        # Pass through encoder layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attention_weights.append(attn_weights)
        
        return x, all_attention_weights

if __name__ == "__main__":
    # Sample sentences
    sentences = [
        "The cat sat on the mat",
        "I love machine learning",
        "Transformers are powerful models"
    ]
    
    # Tokenization
    all_words = ' '.join(sentences).lower().split()
    vocab = sorted(set(all_words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"Vocabulary: {vocab}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Tokenize and pad
    tokenized = []
    max_len = 0
    for sent in sentences:
        tokens = [word_to_idx[w] for w in sent.lower().split()]
        tokenized.append(tokens)
        max_len = max(max_len, len(tokens))
    
    padded_sequences = []
    for tokens in tokenized:
        padded = tokens + [0] * (max_len - len(tokens))
        padded_sequences.append(padded)
    
    input_tensor = torch.tensor(padded_sequences)
    print(f"\nInput tokens:\n{input_tensor}")
    
    # Initialize model
    model = MiniTransformerEncoder(vocab_size, d_model=128, num_heads=4, 
                                   num_layers=2, d_ff=512)
    
    # Forward pass
    output, attention_weights = model(input_tensor)
    
    print(f"\nFinal contextual embeddings shape: {output.shape}")
    print(f"\nAttention heatmap (first sentence, first head):")
    print(attention_weights[0][0, 0].detach().numpy())