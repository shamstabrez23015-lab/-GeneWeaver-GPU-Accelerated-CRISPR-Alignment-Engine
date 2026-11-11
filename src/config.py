"""Shared project configuration for GeneWeaver."""

PROJECT_NAME = "GeneWeaver"

# Valid DNA nucleotide bases (including 'N' for undetermined/any base)
VALID_DNA_BASES = frozenset("ATGCN")

# Default chunk size for processing large genomic sequences (number of base pairs)
DEFAULT_CHUNK_SIZE = 100000

# Default maximum number of mismatches allowed for CRISPR alignment
DEFAULT_MAX_MISMATCHES = 3

