def explain_password(features, predicted_label):
    """
    Build a human-friendly explanation based on extracted features.
    features = dict from extract_features()
    predicted_label = 0/1/2
    """

    explanation = []

    length = features.get("length", 0)
    upper = features.get("upper", 0)
    lower = features.get("lower", 0)
    digit = features.get("digit", 0)
    special = features.get("special", 0)
    entropy = features.get("entropy", 0.0)
    has_common = features.get("has_common", 0)
    has_sequence = features.get("has_sequence", 0)
    has_dict_word = features.get("has_dictionary_word", 0)
    longest_seq = features.get("longest_sequence", 0)

    # -------------------------------
    # Weak password feedback
    # -------------------------------
    if predicted_label == 0:
        explanation.append("Your password is weak. Here’s how you can improve it:")

        if length < 8:
            explanation.append("• Increase length — aim for at least 10–12 characters.")

        if upper == 0:
            explanation.append("• Add uppercase letters to increase complexity.")

        if lower == 0:
            explanation.append("• Add lowercase letters.")

        if digit == 0:
            explanation.append("• Include numbers to strengthen the password.")

        if special == 0:
            explanation.append("• Add special symbols like !, ?, %, @, etc.")

        if entropy < 2:
            explanation.append("• Increase randomness — avoid repeating characters.")

        if has_sequence:
            explanation.append(f"• Avoid sequences such as '123' or 'abc' (detected sequence length = {longest_seq}).")

        if has_dict_word:
            explanation.append("• Remove common words or names from the password.")

        if has_common:
            explanation.append("• Avoid predictable patterns like 'password', '123', etc.")

        if len(explanation) == 1:
            explanation.append("• Try creating a mix of symbols, numbers, and letters.")

        return "\n".join(explanation)

    # -------------------------------
    # Medium password feedback
    # -------------------------------
    if predicted_label == 1:
        explanation.append("Your password is moderately strong. You can make it stronger by:")

        if upper == 0:
            explanation.append("• Adding at least one uppercase letter.")

        if special == 0:
            explanation.append("• Including a special character (e.g., !, %, $, @).")

        if entropy < 3:
            explanation.append("• Increasing randomness and avoiding predictable patterns.")

        if has_sequence:
            explanation.append(f"• Removing sequences like '123' or 'abc' (found sequence length {longest_seq}).")

        if has_dict_word:
            explanation.append("• Avoiding dictionary words or names.")

        return "\n".join(explanation)

    # -------------------------------
    # Strong password feedback
    # -------------------------------
    if predicted_label == 2:
        explanation.append("Your password is strong! Great job.")

        # Optional suggestions to make 'very strong'
        improvements = []

        if upper == 0:
            improvements.append("adding at least one uppercase letter")

        if special == 0:
            improvements.append("including a special symbol")

        if entropy < 3.5:
            improvements.append("making the pattern more random")

        if improvements:
            explanation.append("You can make it even stronger by " + ", ".join(improvements) + ".")

        return "\n".join(explanation)
