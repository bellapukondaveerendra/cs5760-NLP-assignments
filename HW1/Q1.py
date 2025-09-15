import re

# Test data for demonstration
test_text = """
Here are some ZIP codes: 12345, 90210-1234, 78701 5678, A12345 (invalid).
Some words: Hello world, don't worry, state-of-the-art solution, JavaScript.
Numbers: +123, -45.67, 1,234,567.89, 2.5e-10, 42, -3.14e+5.
Contact via email, e-mail, or e mail. Send to EMAIL@example.com.
Reactions: go, goo!, gooo?, goooooo.
Questions end here?
"Is this a question?"
What about this one?'
Not a question.
"""

def test_regex_patterns():
    """Test all 6 regex patterns with sample data."""
    
    # 1. U.S. ZIP codes (disjunction + token boundaries)
    zip_pattern = r'\b\d{5}(?:[-\s]\d{4})?\b'
    print("1. ZIP CODES:")
    zip_matches = re.findall(zip_pattern, test_text)
    for match in zip_matches:
        print(f"   Found: {match}")
    print()
    
    # 2. Words that do NOT start with capital letter
    non_capital_pattern = r"\b[^A-Z\s][^\s]*(?:['-][^\s]+)*\b"
    print("2. NON-CAPITAL WORDS:")
    non_capital_matches = re.findall(non_capital_pattern, test_text)
    for match in non_capital_matches[:10]:  # Show first 10
        print(f"   Found: {match}")
    print(f"   ... and {len(non_capital_matches) - 10} more")
    print()
    
    # 3. Numbers with optional features
    number_pattern = r'[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
    print("3. NUMBERS:")
    number_matches = re.findall(number_pattern, test_text)
    for match in number_matches:
        print(f"   Found: {match}")
    print()
    
    # 4. Email spelling variants (case-insensitive)
    email_pattern = r'\be[-\s–]?mail\b'
    print("4. EMAIL VARIANTS:")
    email_matches = re.findall(email_pattern, test_text, re.IGNORECASE)
    for match in email_matches:
        print(f"   Found: {match}")
    print()
    
    # 5. "go" interjection with optional punctuation
    go_pattern = r'\bgo+\b[!.,?]?'
    print("5. GO INTERJECTIONS:")
    go_matches = re.findall(go_pattern, test_text, re.IGNORECASE)
    for match in go_matches:
        print(f"   Found: '{match}'")
    print()
    
    # 6. Lines ending with question marks (possibly with quotes/brackets)
    question_pattern = r'.*\?[\s"\')\]]*$'
    print("6. QUESTION LINES:")
    lines = test_text.strip().split('\n')
    for line in lines:
        if re.match(question_pattern, line):
            print(f"   Found: {line.strip()}")
    print()

if __name__ == "__main__":
    # Run all examples
    test_regex_patterns()
    
    print("\n" + "="*60)
    print("SUMMARY OF PATTERNS:")
    print("="*60)
    patterns = {
        "ZIP codes": r'\b\d{5}(?:[-\s]\d{4})?\b',
        "Non-capital words": r"\b[^A-Z\s][^\s]*(?:['-][^\s]+)*\b",
        "Numbers": r'[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?',
        "Email variants": r'\be[-\s–]?mail\b',
        "Go interjections": r'\bgo+\b[!.,?]?',
        "Question lines": r'.*\?[\s"\')\]]*$'
    }
    
    for name, pattern in patterns.items():
        print(f"{name:18}: {pattern}")