import re
import numpy as np
from math import log2

# -----------------------------
# LIST OF FEATURES (ORDER MATTERS!)
# -----------------------------
FEATURE_LIST = [
    "length",
    "upper",
    "lower",
    "digit",
    "special",
    "has_common",
    "has_upper_start",
    "ends_with_digit",
    "has_repeating_chars",
    "num_unique_chars",
    "char_type_diversity",
    "entropy",
    "has_sequence",          # NEW FEATURE
    "longest_sequence"       # NEW FEATURE
]

# -----------------------------
# REGEX PATTERNS
# -----------------------------
COMMON_PATTERNS = re.compile(r"(123|password|admin|qwerty|abc|letmein|welcome|iloveyou)")
REPEATING_PATTERN = re.compile(r"(.)\1\1")  # any 3 repeating chars


# -----------------------------
# ENTROPY CALCULATION
# -----------------------------
def calculate_entropy(password: str) -> float:
    if not password:
        return 0.0

    length = len(password)
    freq = {}

    for char in password:
        freq[char] = freq.get(char, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * log2(p)

    return round(entropy, 4)


# -----------------------------
# SEQUENCE DETECTION
# -----------------------------

KEYBOARD_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
]

def detect_sequences(password: str):
    password_lower = password.lower()
    longest = 0
    has_sequence = 0

    # Helper: check if 3+ chars form an increasing/decreasing sequence
    def is_sequence(a, b):
        return ord(b) - ord(a) == 1 or ord(b) - ord(a) == -1

    # 1. Check alphabetic and numeric sequences
    for i in range(len(password_lower) - 1):
        seq_len = 1

        while (
            i + seq_len < len(password_lower)
            and is_sequence(password_lower[i + seq_len - 1], password_lower[i + seq_len])
        ):
            seq_len += 1

        if seq_len >= 3:
            has_sequence = 1
            longest = max(longest, seq_len)

    # 2. Check keyboard sequences like qwerty, asdf, zxcv
    for row in KEYBOARD_ROWS:
        for i in range(len(row) - 2):
            seq = row[i:i+3]
            if seq in password_lower:
                has_sequence = 1
                longest = max(longest, 3)

    return has_sequence, longest


# -----------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_features(password: str) -> dict:
    if not isinstance(password, str):
        return {f: 0 for f in FEATURE_LIST}

    chars = list(password)

    length = len(password)
    upper = sum(c.isupper() for c in chars)
    lower = sum(c.islower() for c in chars)
    digit = sum(c.isdigit() for c in chars)
    special = sum(not c.isalnum() for c in chars)

    has_common = int(bool(COMMON_PATTERNS.search(password.lower())))
    has_upper_start = int(password[0].isupper()) if length > 0 else 0
    ends_with_digit = int(password[-1].isdigit()) if length > 0 else 0
    has_repeating = int(bool(REPEATING_PATTERN.search(password)))
    num_unique_chars = len(set(password))

    char_type_diversity = sum([
        upper > 0,
        lower > 0,
        digit > 0,
        special > 0
    ])

    entropy = calculate_entropy(password)

    # 🔥 NEW: Sequence detection
    has_sequence, longest_sequence = detect_sequences(password)

    return {
        "length": length,
        "upper": upper,
        "lower": lower,
        "digit": digit,
        "special": special,
        "has_common": has_common,
        "has_upper_start": has_upper_start,
        "ends_with_digit": ends_with_digit,
        "has_repeating_chars": has_repeating,
        "num_unique_chars": num_unique_chars,
        "char_type_diversity": char_type_diversity,
        "entropy": entropy,
        "has_sequence": has_sequence,
        "longest_sequence": longest_sequence
    }


# -----------------------------
# MATRIX BUILDER
# -----------------------------
def build_feature_matrix(passwords):
    matrix = []
    for pw in passwords:
        feats = extract_features(pw)
        ordered_feats = [feats[f] for f in FEATURE_LIST]
        matrix.append(ordered_feats)

    return np.array(matrix)
