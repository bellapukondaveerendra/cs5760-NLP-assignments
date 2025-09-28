from collections import defaultdict, Counter
import math

class BigramLanguageModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocabulary = set()
    
    def train(self, corpus):
        """
        Train the bigram language model on a corpus
        
        Args:
            corpus: List of sentences (each sentence is a string)
        """
        for sentence in corpus:
            # Tokenize sentence
            tokens = sentence.strip().split()
            
            # Add tokens to vocabulary
            self.vocabulary.update(tokens)
            
            # Count unigrams
            for token in tokens:
                self.unigram_counts[token] += 1
            
            # Count bigrams
            for i in range(len(tokens) - 1):
                prev_word = tokens[i]
                next_word = tokens[i + 1]
                self.bigram_counts[prev_word][next_word] += 1
    
    def get_unigram_probability(self, word):
        """Calculate unigram probability using MLE"""
        total_words = sum(self.unigram_counts.values())
        return self.unigram_counts[word] / total_words if total_words > 0 else 0
    
    def get_bigram_probability(self, prev_word, word):
        """Calculate bigram probability using MLE"""
        prev_word_count = self.unigram_counts[prev_word]
        if prev_word_count == 0:
            return 0
        return self.bigram_counts[prev_word][word] / prev_word_count
    
    def calculate_sentence_probability(self, sentence):
        """
        Calculate the probability of a sentence using the bigram model
        
        Args:
            sentence: String representing the sentence
            
        Returns:
            Probability of the sentence
        """
        tokens = sentence.strip().split()
        
        if len(tokens) < 2:
            return 0
        
        # Start with probability 1
        probability = 1.0
        
        # Calculate bigram probabilities
        for i in range(len(tokens) - 1):
            prev_word = tokens[i]
            next_word = tokens[i + 1]
            bigram_prob = self.get_bigram_probability(prev_word, next_word)
            
            # If any bigram has probability 0, entire sentence has probability 0
            if bigram_prob == 0:
                return 0
            
            probability *= bigram_prob
        
        return probability
    
    def print_model_stats(self):
        """Print statistics about the trained model"""
        print("=== MODEL STATISTICS ===")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Total unigram counts: {sum(self.unigram_counts.values())}")
        print(f"Number of unique bigrams: {sum(len(bigrams) for bigrams in self.bigram_counts.values())}")
        
        print("\n=== UNIGRAM COUNTS ===")
        for word, count in sorted(self.unigram_counts.items()):
            print(f"{word}: {count}")
        
        print("\n=== BIGRAM COUNTS ===")
        for prev_word in sorted(self.bigram_counts.keys()):
            for next_word, count in sorted(self.bigram_counts[prev_word].items()):
                print(f"{prev_word} → {next_word}: {count}")
        
        print("\n=== BIGRAM PROBABILITIES ===")
        for prev_word in sorted(self.bigram_counts.keys()):
            for next_word in sorted(self.bigram_counts[prev_word].keys()):
                prob = self.get_bigram_probability(prev_word, next_word)
                print(f"P({next_word}|{prev_word}) = {prob:.3f}")

def main():
    # Training corpus from the problem
    training_corpus = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>",
        "<s> deep learning is fun </s>"
    ]
    
    print("=== TRAINING CORPUS ===")
    for i, sentence in enumerate(training_corpus, 1):
        print(f"{i}. {sentence}")
    
    # Create and train the model
    model = BigramLanguageModel()
    model.train(training_corpus)
    
    # Print model statistics
    print()
    model.print_model_stats()
    
    # Test sentences
    test_sentences = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>"
    ]
    
    print("\n" + "="*50)
    print("=== SENTENCE PROBABILITY CALCULATION ===")
    
    sentence_probs = {}
    
    for sentence in test_sentences:
        prob = model.calculate_sentence_probability(sentence)
        sentence_probs[sentence] = prob
        
        print(f"\nSentence: {sentence}")
        print(f"Probability: {prob:.6f}")
        
        # Show step-by-step calculation
        tokens = sentence.strip().split()
        print("Step-by-step calculation:")
        running_prob = 1.0
        
        for i in range(len(tokens) - 1):
            prev_word = tokens[i]
            next_word = tokens[i + 1]
            bigram_prob = model.get_bigram_probability(prev_word, next_word)
            running_prob *= bigram_prob
            print(f"  P({next_word}|{prev_word}) = {bigram_prob:.3f}, Running probability = {running_prob:.6f}")
    
    # Determine which sentence is more probable
    print("\n" + "="*50)
    print("=== CONCLUSION ===")
    
    best_sentence = max(sentence_probs.keys(), key=sentence_probs.get)
    best_prob = sentence_probs[best_sentence]
    
    print(f"Most probable sentence: {best_sentence}")
    print(f"Probability: {best_prob:.6f}")
    
    print("\nReason:")
    print("The model prefers this sentence because it has higher probability")
    print("under the maximum likelihood estimation from the training corpus.")
    
    # Show probability comparison
    print(f"\nProbability comparison:")
    for sentence, prob in sentence_probs.items():
        print(f"  {sentence}: {prob:.6f}")

if __name__ == "__main__":
    main()