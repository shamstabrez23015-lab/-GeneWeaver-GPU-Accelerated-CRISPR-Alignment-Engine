"""CPU-based sequence alignment module for GeneWeaver."""

try:
    from src.config import DEFAULT_MAX_MISMATCHES, VALID_DNA_BASES
    from src.models import AlignmentMatch
except (ImportError, ModuleNotFoundError):
    from config import DEFAULT_MAX_MISMATCHES, VALID_DNA_BASES
    from models import AlignmentMatch


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


def find_off_targets(
    genome_sequence: str,
    guide_sequence: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> list[AlignmentMatch]:
    """Find potential CRISPR off-target alignment sites in a genome sequence.

    Scans the genome sequence using a sliding window of the same length as the
    guide sequence and identifies all windows with a number of mismatched bases
    less than or equal to max_mismatches.

    Args:
        genome_sequence: Target genomic DNA sequence.
        guide_sequence: Query CRISPR guide RNA / DNA sequence.
        max_mismatches: Maximum number of allowable mismatches (default: 3).

    Returns:
        A list of AlignmentMatch objects in genomic order.

    Raises:
        ValueError: If max_mismatches is negative, guide_sequence is empty,
                    or either sequence contains invalid DNA characters.
    """
    if max_mismatches < 0:
        raise ValueError(f"max_mismatches must be non-negative, got {max_mismatches}.")

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    if genome_len < guide_len:
        return []

    matches: list[AlignmentMatch] = []

    for start in range(genome_len - guide_len + 1):
        end = start + guide_len
        window = cleaned_genome[start:end]

        mismatch_positions = [
            j for j in range(guide_len) if window[j] != cleaned_guide[j]
        ]
        mismatch_count = len(mismatch_positions)

        if mismatch_count <= max_mismatches:
            matches.append(
                AlignmentMatch(
                    start=start,
                    end=end,
                    matched_sequence=window,
                    mismatch_count=mismatch_count,
                    mismatch_positions=mismatch_positions,
                )
            )

    return matches


