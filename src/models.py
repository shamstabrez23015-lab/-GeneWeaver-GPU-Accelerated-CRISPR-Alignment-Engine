"""Data models for GeneWeaver."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AlignmentMatch:
    """Represents a matched genomic sequence with alignment and mismatch details.

    Attributes:
        start: Starting index (0-based) of the alignment in the target sequence.
        end: Ending index (exclusive) of the alignment in the target sequence.
        matched_sequence: The genomic nucleotide sequence that was matched.
        mismatch_count: Total number of base mismatches between query and target.
        mismatch_positions: List of indices indicating where mismatches occurred.
    """

    start: int
    end: int
    matched_sequence: str
    mismatch_count: int
    mismatch_positions: List[int] = field(default_factory=list)

