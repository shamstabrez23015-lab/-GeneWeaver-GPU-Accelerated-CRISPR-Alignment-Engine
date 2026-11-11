"""CPU-based sequence alignment module for GeneWeaver."""

try:
    from src.config import VALID_DNA_BASES
except (ImportError, ModuleNotFoundError):
    from config import VALID_DNA_BASES


def validate_dna_sequence(sequence: str) -> str:
    """Validate and clean a DNA sequence string.

    Removes all whitespace, converts the sequence to uppercase, and verifies
    that all characters belong to the allowed set of DNA bases (A, T, G, C, N).

    Args:
        sequence: The DNA sequence string to validate.

    Returns:
        The cleaned, uppercase DNA sequence.

    Raises:
        ValueError: If any invalid DNA characters are detected.
    """
    # Remove all whitespace characters (spaces, tabs, newlines)
    cleaned = "".join(sequence.split()).upper()

    # Detect any characters that are not in VALID_DNA_BASES
    invalid_chars = set(cleaned) - VALID_DNA_BASES
    if invalid_chars:
        sorted_invalid = sorted(invalid_chars)
        raise ValueError(
            f"Invalid DNA character(s) found: {sorted_invalid}. "
            f"Allowed bases are: {sorted(VALID_DNA_BASES)}"
        )

    return cleaned

