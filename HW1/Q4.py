import numpy as np

def edit_distance(s1, s2, sub_cost=1, ins_cost=1, del_cost=1):
    """Compute minimum edit distance with DP"""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i * del_cost
    for j in range(n + 1):
        dp[0][j] = j * ins_cost
    
    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                substitute = dp[i-1][j-1] + sub_cost
                insert = dp[i][j-1] + ins_cost  
                delete = dp[i-1][j] + del_cost
                dp[i][j] = min(substitute, insert, delete)
    
    return dp[m][n], dp

def get_alignment(s1, s2, dp, sub_cost=1, ins_cost=1, del_cost=1):
    """Backtrack to get one valid edit sequence"""
    i, j = len(s1), len(s2)
    operations = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1] and dp[i][j] == dp[i-1][j-1]:
            operations.append(f"Match '{s1[i-1]}'")
            i, j = i-1, j-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + sub_cost:
            operations.append(f"Substitute '{s1[i-1]}' → '{s2[j-1]}'")
            i, j = i-1, j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + ins_cost:
            operations.append(f"Insert '{s2[j-1]}'")
            j = j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + del_cost:
            operations.append(f"Delete '{s1[i-1]}'")
            i = i-1
    
    return list(reversed(operations))

# Q4: Sunday → Saturday
print("Q4: Edit Distance - Sunday → Saturday")
print("=" * 40)

s1, s2 = "Sunday", "Saturday"

# Model A: Sub=1, Ins=1, Del=1
print("\nModel A (Sub=1, Ins=1, Del=1):")
dist_a, matrix_a = edit_distance(s1, s2, 1, 1, 1)
alignment_a = get_alignment(s1, s2, matrix_a, 1, 1, 1)

print(f"Minimum edit distance: {dist_a}")
print("Edit sequence:")
for i, op in enumerate(alignment_a, 1):
    print(f"  {i}. {op}")

# Model B: Sub=2, Ins=1, Del=1  
print("\nModel B (Sub=2, Ins=1, Del=1):")
dist_b, matrix_b = edit_distance(s1, s2, 2, 1, 1)
alignment_b = get_alignment(s1, s2, matrix_b, 2, 1, 1)

print(f"Minimum edit distance: {dist_b}")
print("Edit sequence:")
for i, op in enumerate(alignment_b, 1):
    print(f"  {i}. {op}")

# Reflection
print("\nReflection:")
print("-" * 20)

same_distance = "Yes" if dist_a == dist_b else "No"
print(f"1. Same distance? {same_distance} (A={dist_a}, B={dist_b})")

print("2. Most useful operations: Substitutions and insertions")
print("   - Many character mismatches between the words")  
print("   - Saturday is longer, requiring insertions")

print("3. Application effects:")
print("   - Spell check: Model A better (substitutions common in typos)")
print("   - DNA alignment: Model B better (insertions/deletions more natural)")