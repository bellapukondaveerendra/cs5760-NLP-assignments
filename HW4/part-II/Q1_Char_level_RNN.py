import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, rnn_type='LSTM'):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

def prepare_data(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def create_sequences(text, char_to_idx, seq_length=50):
    sequences = []
    targets = []
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]
        target = text[i+1:i+seq_length+1]
        sequences.append([char_to_idx[ch] for ch in seq])
        targets.append([char_to_idx[ch] for ch in target])
    return torch.tensor(sequences), torch.tensor(targets)

def train_char_rnn(model, train_seq, train_tgt, val_seq, val_tgt, epochs=20, batch_size=64, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        num_batches = len(train_seq) // batch_size
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            
            batch_seq = train_seq[batch_start:batch_end]
            batch_target = train_tgt[batch_start:batch_end]
            
            optimizer.zero_grad()
            output, _ = model(batch_seq)
            
            loss = criterion(output.view(-1, output.size(-1)), batch_target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output, _ = model(val_seq)
            val_loss = criterion(val_output.view(-1, val_output.size(-1)), val_tgt.view(-1))
            val_losses.append(val_loss.item())
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
    
    return train_losses, val_losses

def generate_text(model, start_text, char_to_idx, idx_to_char, length=300, temperature=1.0):
    model.eval()
    
    with torch.no_grad():
        current_seq = [char_to_idx[ch] for ch in start_text]
        generated = start_text
        
        hidden = None
        
        for _ in range(length):
            x = torch.tensor([current_seq[-50:]]).long()
            output, hidden = model(x, hidden)
            
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            
            generated += next_char
            current_seq.append(next_idx)
        
        return generated

if __name__ == "__main__":
    # Load or create corpus
    # For demo, using toy corpus - replace with 50-200KB text file
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # If file doesn't exist, use toy corpus
    if len(text) < 100:
        text = """hello world. help me learn. hello there. 
how are you today. hope you are well. hello friend.
learning is fun. today is great. """ * 50
    
    print(f"Corpus length: {len(text)} characters")
    
    char_to_idx, idx_to_char, vocab_size = prepare_data(text)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split into train/validation
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create sequences
    seq_length = 50
    train_seq, train_tgt = create_sequences(train_text, char_to_idx, seq_length)
    val_seq, val_tgt = create_sequences(val_text, char_to_idx, seq_length)
    
    print(f"Training sequences: {len(train_seq)}")
    print(f"Validation sequences: {len(val_seq)}")
    
    # Initialize model
    hidden_size = 128
    model = CharRNN(vocab_size, embedding_dim=64, hidden_size=hidden_size, rnn_type='LSTM')
    
    # Train
    train_losses, val_losses = train_char_rnn(
        model, train_seq, train_tgt, val_seq, val_tgt, 
        epochs=20, batch_size=64, lr=0.001
    )
    
    # 1. Plot training/validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nLoss curves saved to 'loss_curves.png'")
    
    # 2. Generate samples with different temperatures
    print("\n" + "="*70)
    print("TEMPERATURE-CONTROLLED GENERATIONS")
    print("="*70)
    
    temperatures = [0.7, 1.0, 1.2]
    
    for temp in temperatures:
        print(f"\n{'='*70}")
        print(f"Temperature τ = {temp}")
        print(f"{'='*70}")
        generated = generate_text(model, "hello", char_to_idx, idx_to_char, 
                                  length=300, temperature=temp)
        print(generated)
    
    # 3. Reflection (3-5 sentences)
    print("\n" + "="*70)
    print("REFLECTION: Impact of Hyperparameters")
    print("="*70)
    
    reflection = """
Sequence Length: Longer sequences (e.g., 100 vs 50) allow the model to capture 
longer-range dependencies but require more memory and computation. Shorter sequences 
train faster but may miss important context, leading to less coherent generations.

Hidden Size: Larger hidden dimensions (e.g., 256 vs 128) increase model capacity and 
allow learning more complex patterns, but risk overfitting on small datasets and 
require more training time. Smaller hidden sizes generalize better but may underfit.

Temperature: Low temperature (τ < 1.0) makes sampling more deterministic, producing 
conservative but coherent text by favoring high-probability tokens. High temperature 
(τ > 1.0) increases randomness and diversity but may generate nonsensical output. 
Temperature τ = 1.0 provides balanced exploration of the learned distribution.
    """
    
    print(reflection)
    
    # Save model
    torch.save(model.state_dict(), 'char_rnn_model.pt')
    print("\nModel saved to 'char_rnn_model.pt'")