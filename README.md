# CS5760 Natural Language Processing - Homework 1

- **Name:** Veerendra Bellapukonda
- **StudentId:** 700772385
- **Course:** CS5760 Natural Language Processing

## Overview

This homework covers four fundamental NLP concepts:
1. **Regular Expressions** - Pattern matching for text processing
2. **Tokenization** - Breaking text into meaningful units
3. **Byte Pair Encoding (BPE)** - Subword tokenization algorithm
4. **Edit Distance** - Computing minimum transformation cost between strings

## Question 1: Regular Expressions (Q1.py)

**Patterns Implemented:**
- **ZIP Codes:** `\b\d{5}(?:[-\s]\d{4})?\b` - Matches US ZIP codes with optional +4 extension
- **Non-Capital Words:** `\b[^A-Z\s][^\s]*(?:['-][^\s]+)*\b` - Words not starting with capitals
- **Rich Numbers:** `[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?` - Numbers with signs, commas, decimals, scientific notation
- **Email Variants:** `\be[-\s–]?mail\b` - Matches "email", "e-mail", "e mail" (case-insensitive)
- **Go Interjections:** `\bgo+\b[!.,?]?` - Matches "go", "goo", "gooo" with optional punctuation
- **Question Lines:** `.*\?[\s"')\]]*$` - Lines ending with question marks and optional quotes/brackets

**Usage:** Run `python Q1.py` to test all patterns with sample data.

## Question 2: Tokenization (Q2.py)

**Components:**
- **Naive Tokenization:** Simple space-based splitting
- **Manual Tokenization:** Handles contractions, punctuation, possessives
- **Tool Comparison:** Uses NLTK and spaCy tokenizers
- **MWE Analysis:** Identifies multiword expressions in tech domain

**Sample Text:** Technology news paragraph about AI startups and NLP

**Key Findings:**
- Manual tokenization expands contractions ("wasn't" → "was not")
- Tools handle edge cases better than manual rules
- MWEs like "natural language processing" should be single tokens

**Dependencies:**
```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```

## Question 3: Byte Pair Encoding (Q3.py)

**Three Parts:**

### 3.1 Manual BPE on Toy Corpus
- Corpus: "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
- Performs first 3 BPE merges by hand
- Shows vocabulary growth step-by-step

### 3.2 Coded BPE Learner
- Complete BPE implementation with `BPELearner` class
- Tests on words: "new", "newer", "lowest", "widest", "newestest"
- Demonstrates OOV handling and morpheme alignment

### 3.3 BPE on English Paragraph
- Trains on NLP-related text with 30 merges
- Segments 5 test words including rare and derived forms
- Analyzes learned prefixes, suffixes, and stems

**Key Insights:**
- BPE solves OOV problems through subword composition
- Discovers morphological boundaries (e.g., "er_" suffix)
- Balances vocabulary size with representation power

## Question 4: Edit Distance (Q4.py)

**Task:** Compute minimum edit distance "Sunday" → "Saturday"

**Two Models:**
- **Model A:** Substitution=1, Insertion=1, Deletion=1
- **Model B:** Substitution=2, Insertion=1, Deletion=1

**Implementation:**
- Dynamic programming with DP matrix
- Backtracking for optimal alignment sequences
- Comparison of operation preferences between models

**Applications:**
- Spell checking favors Model A (equal operation costs)
- DNA alignment might prefer Model B (expensive substitutions)

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install numpy nltk spacy
   python -m spacy download en_core_web_sm
   ```

2. **Run Individual Questions:**
   ```bash
   python Q1.py  # Regular expressions
   python Q2.py  # Tokenization
   python Q3.py  # BPE
   python Q4.py  # Edit distance
   ```


