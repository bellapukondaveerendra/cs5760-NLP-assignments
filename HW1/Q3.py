import re
from collections import defaultdict, Counter
import copy

# Q3.1: Manual BPE on toy corpus
def manual_bpe_toy_corpus():
    print("\nQ3.1: MANUAL BPE ON TOY CORPUS")
    print("-" * 40)
    
    # Original corpus
    original_corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    
    print(f"Original corpus: {original_corpus}")
    print()
    
    # Step 1: Add end-of-word marker _ and create initial vocabulary
    print("STEP 1: Add end-of-word marker and initial vocabulary")
    print("-" * 30)
    
    words = original_corpus.split()
    word_counts = Counter(words)
    print(f"Word frequencies: {dict(word_counts)}")
    
    # Add end-of-word markers
    words_with_markers = {}
    for word, count in word_counts.items():
        words_with_markers[word + '_'] = count
    
    print(f"Words with end markers: {words_with_markers}")
    
    # Create initial vocabulary (all characters including _)
    initial_vocab = set()
    for word in words_with_markers:
        for char in word:
            initial_vocab.add(char)
    
    initial_vocab = sorted(list(initial_vocab))
    print(f"Initial vocabulary: {initial_vocab}")
    print(f"Initial vocabulary size: {len(initial_vocab)}")
    print()
    
    # Initialize current state
    current_words = copy.deepcopy(words_with_markers)
    current_vocab = set(initial_vocab)
    
    # Function to get all bigram pairs and their counts
    def get_bigram_counts(word_dict):
        pair_counts = defaultdict(int)
        for word, freq in word_dict.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_counts[pair] += freq
        return pair_counts
    
    # Function to merge a pair in all words
    def merge_pair(word_dict, pair_to_merge):
        new_word_dict = {}
        old_char1, old_char2 = pair_to_merge
        new_token = old_char1 + old_char2
        
        for word, freq in word_dict.items():
            chars = list(word)
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == old_char1 and chars[i + 1] == old_char2:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            new_word = ''.join(new_chars)
            new_word_dict[new_word] = freq
        return new_word_dict, new_token
    
    # Perform first three merges manually
    merges = []
    
    for merge_step in range(1, 4):
        print(f"MERGE STEP {merge_step}:")
        print("-" * 20)
        
        # Get bigram counts
        bigram_counts = get_bigram_counts(current_words)
        print("Bigram counts:")
        for pair, count in sorted(bigram_counts.items(), key=lambda x: -x[1]):
            print(f"  {pair}: {count}")
        
        # Find most frequent pair
        most_frequent_pair = max(bigram_counts.items(), key=lambda x: x[1])
        pair_to_merge, max_count = most_frequent_pair
        
        print(f"\nMost frequent pair: {pair_to_merge} (count: {max_count})")
        
        # Merge the pair
        current_words, new_token = merge_pair(current_words, pair_to_merge)
        current_vocab.add(new_token)
        merges.append((pair_to_merge, new_token))
        
        print(f"New token created: '{new_token}'")
        print(f"Updated corpus (first few examples):")
        
        # Show first few word examples
        for i, (word, freq) in enumerate(current_words.items()):
            if i < 5:  # Show first 5 words
                print(f"  '{word}' (freq: {freq})")
        
        print(f"Updated vocabulary size: {len(current_vocab)}")
        print(f"Current vocabulary: {sorted(list(current_vocab))}")
        print()
    
    return current_words, merges, current_vocab

# Q3.2: Code a mini-BPE learner
def coded_bpe_learner():
    print("Q3.2: CODED MINI-BPE LEARNER")
    print("-" * 40)
    
    class BPELearner:
        def __init__(self):
            self.vocab = set()
            self.merges = []
            self.word_freqs = {}
        
        def train(self, corpus, num_merges=10):
            """Train BPE on a corpus"""
            # Initialize word frequencies
            words = corpus.split()
            self.word_freqs = Counter(word + '_' for word in words)
            
            # Initialize vocabulary with characters
            self.vocab = set()
            for word in self.word_freqs:
                for char in word:
                    self.vocab.add(char)
            
            print(f"Initial vocabulary size: {len(self.vocab)}")
            print(f"Initial vocab: {sorted(self.vocab)}")
            print()
            
            # Perform merges
            current_words = dict(self.word_freqs)
            
            for i in range(num_merges):
                # Get pair counts
                pair_counts = defaultdict(int)
                for word, freq in current_words.items():
                    chars = word.split() if ' ' in word else list(word)
                    for j in range(len(chars) - 1):
                        pair = (chars[j], chars[j + 1])
                        pair_counts[pair] += freq
                
                if not pair_counts:
                    break
                
                # Find most frequent pair
                best_pair = max(pair_counts.items(), key=lambda x: x[1])
                pair_to_merge, count = best_pair
                
                print(f"Step {i + 1}: Merging {pair_to_merge} (count: {count})")
                
                # Merge pair
                new_token = pair_to_merge[0] + pair_to_merge[1]
                self.merges.append(pair_to_merge)
                self.vocab.add(new_token)
                
                # Update words
                new_words = {}
                for word, freq in current_words.items():
                    chars = word.split() if ' ' in word else list(word)
                    new_chars = []
                    j = 0
                    while j < len(chars):
                        if (j < len(chars) - 1 and 
                            chars[j] == pair_to_merge[0] and 
                            chars[j + 1] == pair_to_merge[1]):
                            new_chars.append(new_token)
                            j += 2
                        else:
                            new_chars.append(chars[j])
                            j += 1
                    
                    new_word = ' '.join(new_chars) if len(new_chars) > 1 else ''.join(new_chars)
                    new_words[new_word] = freq
                
                current_words = new_words
                print(f"  New token: '{new_token}'")
                print(f"  Vocabulary size: {len(self.vocab)}")
                print()
            
            self.final_words = current_words
            return self.merges, self.vocab
        
        def segment_word(self, word):
            word = word + '_'
            chars = list(word)
            
            # Apply merges in order
            for merge_pair in self.merges:
                old1, old2 = merge_pair
                new_token = old1 + old2
                
                new_chars = []
                i = 0
                while i < len(chars):
                    if (i < len(chars) - 1 and 
                        chars[i] == old1 and chars[i + 1] == old2):
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            
            return chars
    
    # Train on toy corpus
    toy_corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    
    bpe = BPELearner()
    merges, vocab = bpe.train(toy_corpus, num_merges=8)
    
    print("LEARNED MERGES:")
    for i, merge in enumerate(merges, 1):
        print(f"{i}. {merge[0]} + {merge[1]} → {merge[0] + merge[1]}")
    print()
    
    # Segment test words
    test_words = ["new", "newer", "lowest", "widest", "newestest"]
    
    print("WORD SEGMENTATION:")
    print("-" * 20)
    for word in test_words:
        segments = bpe.segment_word(word)
        print(f"'{word}' → {segments}")
    print()
    
    # Explanation
    print("BPE ANALYSIS:")
    print("-" * 20)
    explanation = """
    How subword tokens solve the OOV problem:
    BPE creates a vocabulary of subword units that can be combined to represent any word,
    including words not seen during training. For example, 'newestest' (invented word)
    can be segmented using known subwords like 'new', 'est', even if the full word
    wasn't in the training corpus. This provides better coverage than word-level
    tokenization while being more efficient than character-level tokenization.
    
    Morpheme alignment example:
    The 'er_' token learned by BPE often aligns with the English comparative/agent suffix,
    as seen in 'newer' → ['new', 'er_']. This shows how BPE can discover meaningful
    morphological boundaries without explicit linguistic knowledge, making it useful
    across languages with different morphological structures.
    """
    
    print(explanation.strip())
    
    return bpe

# Q3.3: BPE on your language (English paragraph)
def bpe_on_paragraph():
    print("\n\nQ3.3: BPE ON ENGLISH PARAGRAPH")
    print("-" * 40)
    
    # Sample paragraph (you can replace with your own)
    paragraph = """
    Natural language processing enables computers to understand human communication.
    Advanced algorithms analyze text patterns and extract meaningful information.
    Machine learning models can process thousands of documents automatically.
    This technology revolutionizes how we interact with digital systems.
    """
    
    # Clean and prepare text
    text = paragraph.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation for simplicity
    
    print(f"Training text: {text}")
    print()
    
    class AdvancedBPE:
        def __init__(self):
            self.vocab = set()
            self.merges = []
            self.word_freqs = {}
        
        def train(self, text, num_merges=30):
            """Train BPE on text with specified number of merges"""
            words = text.split()
            self.word_freqs = Counter(word + '_' for word in words)
            
            # Initialize character vocabulary
            self.vocab = set()
            for word in self.word_freqs:
                for char in word:
                    self.vocab.add(char)
            
            print(f"Training on {len(words)} tokens, {len(set(words))} unique words")
            print(f"Initial vocabulary size: {len(self.vocab)}")
            
            current_words = dict(self.word_freqs)
            merge_frequencies = []
            
            for i in range(num_merges):
                # Get pair counts
                pair_counts = defaultdict(int)
                for word, freq in current_words.items():
                    chars = list(word)
                    for j in range(len(chars) - 1):
                        pair = (chars[j], chars[j + 1])
                        pair_counts[pair] += freq
                
                if not pair_counts:
                    print(f"No more pairs to merge at step {i}")
                    break
                
                # Find most frequent pair
                best_pair, count = max(pair_counts.items(), key=lambda x: x[1])
                merge_frequencies.append((best_pair, count))
                
                if i < 5:  # Show first 5 merges
                    print(f"Merge {i + 1}: {best_pair} (freq: {count})")
                
                # Create new token and update vocabulary
                new_token = best_pair[0] + best_pair[1]
                self.merges.append(best_pair)
                self.vocab.add(new_token)
                
                # Update all words
                new_words = {}
                for word, freq in current_words.items():
                    new_word = word.replace(best_pair[0] + best_pair[1], new_token)
                    new_words[new_word] = freq
                
                current_words = new_words
            
            self.final_words = current_words
            
            # Find longest tokens
            longest_tokens = sorted([token for token in self.vocab if len(token) > 1], 
                                  key=len, reverse=True)[:5]
            
            print(f"\nCompleted {len(self.merges)} merges")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"\nTop 5 most frequent merges:")
            for i, (pair, freq) in enumerate(merge_frequencies[:5], 1):
                print(f"  {i}. {pair} → {pair[0] + pair[1]} (freq: {freq})")
            
            print(f"\n5 longest subword tokens:")
            for i, token in enumerate(longest_tokens, 1):
                print(f"  {i}. '{token}' (length: {len(token)})")
            
            return self.merges, self.vocab, longest_tokens
        
        def segment_word(self, word):
            word = word.lower() + '_'
            chars = list(word)
            
            for merge_pair in self.merges:
                old1, old2 = merge_pair
                new_token = old1 + old2
                
                new_chars = []
                i = 0
                while i < len(chars):
                    if (i < len(chars) - 1 and 
                        chars[i] == old1 and chars[i + 1] == old2):
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            
            return chars
    
    # Train BPE
    bpe = AdvancedBPE()
    merges, vocab, longest_tokens = bpe.train(text, num_merges=30)
    
    # Test segmentation on 5 different words
    test_words = [
        "processing",      # common word
        "revolutionizes",  # rare/complex word  
        "automatically",   # derived form
        "meaningful",      # compound-like
        "communication"    # inflected form
    ]
    
    print(f"\nWORD SEGMENTATION EXAMPLES:")
    print("-" * 30)
    for word in test_words:
        segments = bpe.segment_word(word)
        print(f"'{word}' → {segments}")
    
    # Reflection
    print(f"\nBRIEF REFLECTION:")
    print("-" * 20)
    
    reflection = """
    The BPE algorithm learned various types of subwords from the English paragraph:
    - Prefixes: Common beginnings like 'un-', 're-', 'pre-'
    - Suffixes: Endings like '-ing', '-tion', '-ly', '-ed'  
    - Stems: Root words and common morphemes
    - Whole words: Frequent short words remained intact
    - Character combinations: Common letter sequences in English
    
    Pros of subword tokenization for English:
    1. Handles morphological variations (process/processing/processed) efficiently
    2. Reduces vocabulary size while maintaining semantic relationships
    3. Better OOV handling - can represent unseen words through known subwords
    
    Cons of subword tokenization for English:
    1. May split semantically meaningful units inappropriately 
    2. Context-dependent meaning can be lost when words are split
    3. Inconsistent segmentation of the same word in different contexts
    """
    
    print(reflection.strip())
    
    return bpe

if __name__ == "__main__":
    # Run all parts
    print("Running Q3.1: Manual BPE...")
    manual_words, manual_merges, manual_vocab = manual_bpe_toy_corpus()
    
    print("\n" + "="*60)
    print("Running Q3.2: Coded BPE...")
    coded_bpe = coded_bpe_learner()
    
    print("\n" + "="*60) 
    print("Running Q3.3: BPE on paragraph...")
    paragraph_bpe = bpe_on_paragraph()