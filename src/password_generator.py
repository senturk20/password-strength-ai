"""Password suggestion and generation utilities.

This module provides a policy-driven, cryptographically-secure
password generator and a unified suggestions API. It intentionally
avoids external dependencies and is compatible with Python 3.10+.
"""

import re
import secrets
import string
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -----------------------------
# Policy
# -----------------------------
@dataclass
class PasswordPolicy:
    min_length: int = 12
    max_length: int = 64
    require_upper: bool = True
    require_lower: bool = True
    require_digit: bool = True
    require_special: bool = True
    min_diversity: int = 3
    banned_substrings: Tuple[str, ...] = (
        "password",
        "admin",
        "qwerty",
        "letmein",
        "welcome",
        "iloveyou",
        "123456",
        "111111",
        "000000",
        "abcdef",
    )
    banned_regexes: Tuple[re.Pattern, ...] = (
        re.compile(r"(.)\1\1"),  # 3 repeating chars
        re.compile(r"0123|1234|2345|3456|4567|5678|6789|7890"),
        re.compile(r"9876|8765|7654|6543|5432|4321|3210"),
        re.compile(r"abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz"),
    )

    def check(self, pw: str) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not isinstance(pw, str):
            reasons.append("password must be a string")
            return False, reasons

        if len(pw) < self.min_length:
            reasons.append(f"length < {self.min_length}")
        if len(pw) > self.max_length:
            reasons.append(f"length > {self.max_length}")

        upper = any(c.isupper() for c in pw)
        lower = any(c.islower() for c in pw)
        digit = any(c.isdigit() for c in pw)
        special = any(not c.isalnum() for c in pw)

        diversity = sum([upper, lower, digit, special])
        if diversity < self.min_diversity:
            reasons.append(f"diversity < {self.min_diversity}")

        if self.require_upper and not upper:
            reasons.append("missing uppercase")
        if self.require_lower and not lower:
            reasons.append("missing lowercase")
        if self.require_digit and not digit:
            reasons.append("missing digit")
        if self.require_special and not special:
            reasons.append("missing special")

        low = pw.lower()
        for sub in self.banned_substrings:
            if sub in low:
                reasons.append(f"contains banned substring '{sub}'")
                break

        for rx in self.banned_regexes:
            if rx.search(low):
                reasons.append("contains banned pattern/sequence")
                break

        return (len(reasons) == 0), reasons


# -----------------------------
# Character sets and maps
# -----------------------------
UPPERS = string.ascii_uppercase
LOWERS = string.ascii_lowercase
DIGITS = string.digits
SPECIALS = "!@#$%^&*()-_=+[]{};:,.?/<>"
ALL_CHARS = UPPERS + LOWERS + DIGITS + SPECIALS

LEET_MAP = {
    "a": ["@", "4"],
    "b": ["8"],
    "e": ["3"],
    "g": ["9"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "t": ["7"],
}

# Small safe wordlist for passphrases (expandable)
PASSPHRASE_WORDS = [
    "river", "forest", "sunrise", "shadow", "silver", "ocean", "mountain", "quiet",
    "bright", "planet", "falcon", "tiger", "ember", "cloud", "stone", "meadow",
    "signal", "matrix", "cipher", "safe", "phoenix", "comet", "anchor", "spectrum",
    "breeze", "aurora", "rocket", "orbit", "cosmos", "winter", "summer", "autumn",
    "spring", "honey", "crystal", "mirror", "nomad", "harbor", "thunder", "gold",
    "atlas", "glacier", "lunar", "solar", "zenith", "vertex", "nova", "owl",
    "eagle", "wolf", "panda", "koala", "delta", "sigma", "vector", "kernel",
    "buffer", "dragonfly", "cactus", "coral", "pepper", "hazel", "ivory", "obsidian",
    "opal", "onyx", "jade", "ruby", "sapphire", "flame", "spark", "echo", "pulse",
    "tempo", "horizon", "compass", "garden", "trail", "lantern", "shield", "voyage",
    "island", "valley", "summit", "canyon", "brook", "rain", "snow", "meadow",
]


# -----------------------------
# Helpers
# -----------------------------
def _secure_choice(seq: str) -> str:
    return secrets.choice(seq)


def _shuffle_str(s: str) -> str:
    arr = list(s)
    secrets.SystemRandom().shuffle(arr)
    return "".join(arr)


def _ensure_policy(pw: str, policy: PasswordPolicy) -> str:
    ok, _ = policy.check(pw)
    if ok:
        return pw

    # Add missing categories
    if policy.require_upper and not any(c.isupper() for c in pw):
        pw += _secure_choice(UPPERS)
    if policy.require_lower and not any(c.islower() for c in pw):
        pw += _secure_choice(LOWERS)
    if policy.require_digit and not any(c.isdigit() for c in pw):
        pw += _secure_choice(DIGITS)
    if policy.require_special and not any(not c.isalnum() for c in pw):
        pw += _secure_choice(SPECIALS)

    # Extend to minimum length
    while len(pw) < policy.min_length:
        pw += _secure_choice(ALL_CHARS)

    # Trim if necessary
    if len(pw) > policy.max_length:
        pw = pw[: policy.max_length]

    return pw


# -----------------------------
# High-entropy random password
# -----------------------------
def generate_random_password(length: int = 16, policy: Optional[PasswordPolicy] = None) -> str:
    policy = policy or PasswordPolicy()
    length = max(length, policy.min_length)

    parts: List[str] = []
    if policy.require_upper:
        parts.append(_secure_choice(UPPERS))
    if policy.require_lower:
        parts.append(_secure_choice(LOWERS))
    if policy.require_digit:
        parts.append(_secure_choice(DIGITS))
    if policy.require_special:
        parts.append(_secure_choice(SPECIALS))

    # Fill remaining with secure choices
    while len(parts) < length:
        parts.append(_secure_choice(ALL_CHARS))

    pw = _shuffle_str("".join(parts))
    pw = _ensure_policy(pw, policy)

    # Try a few regenerations if banned patterns persist
    for _ in range(5):
        ok, _ = policy.check(pw)
        if ok:
            return pw
        # regenerate with the same constraints
        parts = [_secure_choice(ALL_CHARS) for _ in range(length)]
        pw = _shuffle_str("".join(parts))

    return pw


# -----------------------------
# Passphrase (Diceware/XKCD-style)
# -----------------------------
def generate_passphrase(num_words: int = 4, separator: str = "-", policy: Optional[PasswordPolicy] = None) -> str:
    policy = policy or PasswordPolicy()

    words = [secrets.choice(PASSPHRASE_WORDS) for _ in range(num_words)]
    # Randomly capitalize one word
    idx = secrets.randbelow(len(words))
    words[idx] = words[idx].capitalize()

    # Add a digit chunk and a special for diversity
    digit_chunk = str(secrets.randbelow(900) + 100)
    special = _secure_choice(SPECIALS)

    pw = separator.join(words) + separator + digit_chunk + special
    pw = _ensure_policy(pw, policy)
    return pw


# -----------------------------
# Mutation-based improvement
# -----------------------------
def mutate_password(user_pw: str, policy: Optional[PasswordPolicy] = None, k: int = 4) -> List[str]:
    policy = policy or PasswordPolicy()
    base = (user_pw or "").strip()
    if not base:
        return []

    suggestions = set()

    def leetify(s: str) -> str:
        out = []
        for ch in s:
            low = ch.lower()
            if low in LEET_MAP and secrets.randbelow(100) < 50:
                out.append(secrets.choice(LEET_MAP[low]))
            else:
                out.append(ch)
        return "".join(out)

    for _ in range(40):
        pw = base

        # apply leet substitutions
        pw = leetify(pw)

        # random capitalization
        chars = list(pw)
        for i in range(len(chars)):
            if chars[i].isalpha() and secrets.randbelow(100) < 30:
                chars[i] = chars[i].upper() if chars[i].islower() else chars[i].lower()
        pw = "".join(chars)

        # remove weak numeric/end sequences
        pw = re.sub(r"(0123|1234|2345|3456|4567|5678|6789|7890|202[0-9])$", "", pw)

        # append random blocks
        pw += _secure_choice(SPECIALS)
        pw += str(secrets.randbelow(9000) + 1000)

        # insert an extra special at a random position
        pos = secrets.randbelow(len(pw) + 1)
        pw = pw[:pos] + _secure_choice(SPECIALS) + pw[pos:]

        # ensure policy and normalize length
        pw = _ensure_policy(pw, policy)

        ok, _ = policy.check(pw)
        if ok and pw.lower() != base.lower():
            suggestions.add(pw)

        if len(suggestions) >= k:
            break

    return list(suggestions)[:k]


# -----------------------------
# Unified suggestions API
# -----------------------------
def suggest_passwords(
    user_pw: str,
    policy: Optional[PasswordPolicy] = None,
    num_mutations: int = 4,
    num_passphrases: int = 2,
    num_randoms: int = 2,
) -> List[str]:
    policy = policy or PasswordPolicy()

    out: List[str] = []
    out.extend(mutate_password(user_pw, policy=policy, k=num_mutations))

    for _ in range(num_passphrases):
        out.append(generate_passphrase(policy=policy))

    for _ in range(num_randoms):
        out.append(generate_random_password(policy=policy))

    final: List[str] = []
    seen = set()
    for pw in out:
        if pw in seen:
            continue
        seen.add(pw)
        ok, _ = policy.check(pw)
        if ok:
            final.append(pw)

    return final
