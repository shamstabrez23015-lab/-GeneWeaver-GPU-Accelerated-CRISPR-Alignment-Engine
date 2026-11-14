"""
genomic_parser.py
-----------------
Day 1, Day 2, Day 3 & Day 4 – GeneWeaver: Genomic Parser Module

Provides FASTA file parsing, sequence extraction, DNA sequence cleaning,
validation, and genomic sequence chunking utilities using BioPython's SeqIO module.
Responsible for reading genomic FASTA records, extracting cleaned DNA sequences,
retrieving sequences by record ID, and partitioning sequences into chunks
ready for downstream processing (alignment, etc.).
"""

from __future__ import annotations

import os
from typing import Any

from Bio import SeqIO

try:
    from config import DEFAULT_CHUNK_SIZE
except ImportError:
    try:
        from src.config import DEFAULT_CHUNK_SIZE  # type: ignore[import-not-found]
    except ImportError:
        DEFAULT_CHUNK_SIZE: int = 10_000

# Allowed DNA nucleotides: A, T, G, C, plus N for ambiguous/unknown bases
VALID_DNA_CHARACTERS: frozenset[str] = frozenset({"A", "T", "G", "C", "N"})


def clean_dna_sequence(sequence: str) -> str:
    """Clean and validate a DNA sequence string.

    Removes spaces, tabs, and newline characters, converts the sequence to
    uppercase, and verifies that only valid DNA characters (A, T, G, C, N)
    are present. N is preserved as an unknown or ambiguous base.

    Parameters
    ----------
    sequence:
        The raw DNA sequence string to clean and validate.

    Returns
    -------
    str
        The cleaned, uppercase DNA sequence.

    Raises
    ------
    ValueError
        If *sequence* is not a string, contains unsupported characters,
        or is empty after removing whitespace.

    Examples
    --------
    >>> clean_dna_sequence("  atgc \\n\\t ttgN  ")
    'ATGCTTAGN'
    """
    if not isinstance(sequence, str):
        raise ValueError(
            f"DNA sequence must be a string, got {type(sequence).__name__}."
        )

    # Remove spaces, tabs, and newline characters (\r, \n)
    cleaned = "".join(c for c in sequence if c not in " \t\n\r")
    cleaned = cleaned.upper()

    if not cleaned:
        raise ValueError("DNA sequence is empty after cleaning.")

    # Validate against allowed DNA characters (A, T, G, C, N)
    invalid_chars = set(cleaned) - VALID_DNA_CHARACTERS
    if invalid_chars:
        sorted_invalid = sorted(invalid_chars)
        raise ValueError(
            f"Unsupported DNA character(s) found in sequence: {sorted_invalid}. "
            "Only A, T, G, C, and N are allowed."
        )

    return cleaned


def validate_dna_sequence(sequence: str) -> bool:
    """Validate a DNA sequence string using :func:`clean_dna_sequence`.

    Parameters
    ----------
    sequence:
        The DNA sequence string to validate.

    Returns
    -------
    bool
        True if the sequence is valid and non-empty.

    Raises
    ------
    ValueError
        If the sequence is empty, not a string, or contains unsupported characters.

    Examples
    --------
    >>> validate_dna_sequence("ATGC")
    True
    """
    clean_dna_sequence(sequence)
    return True


def parse_fasta(file_path: str) -> list[dict[str, Any]]:
    """Parse a FASTA file and return its records as a list of dictionaries.

    Each record is represented as a dictionary with the following keys:

    - ``record_id``   (str): The sequence identifier (text before the first
                             space on the header line, e.g. ``seq1``).
    - ``description`` (str): The full header string, including the identifier
                             (e.g. ``seq1 Human BRCA1 exon 2``).
    - ``sequence``    (str): The cleaned DNA sequence in uppercase containing
                             only valid nucleotides (A, T, G, C, N).

    Parameters
    ----------
    file_path:
        Absolute or relative path to the FASTA file to parse.

    Returns
    -------
    list[dict[str, Any]]
        A list of record dictionaries, one per FASTA entry.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not point to an existing file.
    ValueError
        If the FASTA file exists but contains no sequence records, or if any
        record contains an invalid or empty DNA sequence.

    Examples
    --------
    >>> records = parse_fasta("sample.fasta")
    >>> records[0]["record_id"]
    'seq1'
    >>> records[0]["sequence"]
    'ATGCTTAG'
    """
    # --- file existence check -------------------------------------------------
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"FASTA file not found: '{file_path}'"
        )

    # --- parse ----------------------------------------------------------------
    records: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for seq_record in SeqIO.parse(handle, "fasta"):
            raw_seq = str(seq_record.seq)
            try:
                cleaned_seq = clean_dna_sequence(raw_seq)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid DNA sequence in record '{seq_record.id}': {exc}"
                ) from exc

            records.append(
                {
                    "record_id": seq_record.id,
                    "description": seq_record.description,
                    "sequence": cleaned_seq,
                }
            )

    # --- validate -------------------------------------------------------------
    if not records:
        raise ValueError(
            f"No FASTA records found in file: '{file_path}'. "
            "The file may be empty or malformed."
        )

    return records


def extract_sequences(records: list[dict[str, Any]]) -> list[str]:
    """Extract DNA sequences from a list of FASTA record dictionaries.

    Preserves the original record order and guarantees that all returned
    sequences are cleaned uppercase strings.

    Parameters
    ----------
    records:
        A list of record dictionaries, such as those returned by :func:`parse_fasta`.
        Each dictionary must contain a valid ``'sequence'`` key with a non-empty string.

    Returns
    -------
    list[str]
        A list of cleaned uppercase DNA sequence strings in the same order as *records*.
        Returns an empty list if *records* is empty.

    Raises
    ------
    ValueError
        If any record in *records* is not a dictionary or does not contain a
        valid sequence field.

    Examples
    --------
    >>> records = [{"record_id": "s1", "description": "s1", "sequence": "atgc"}]
    >>> extract_sequences(records)
    ['ATGC']
    >>> extract_sequences([])
    []
    """
    if not records:
        return []

    sequences: list[str] = []
    for record in records:
        if not isinstance(record, dict) or "sequence" not in record:
            raise ValueError(
                f"Record is missing required 'sequence' field: {record}"
            )
        seq = record["sequence"]
        cleaned = clean_dna_sequence(seq)
        sequences.append(cleaned)

    return sequences


def get_sequence_by_id(
    records: list[dict[str, Any]],
    record_id: str,
) -> str:
    """Find a FASTA record by its record_id and return its DNA sequence.

    Parameters
    ----------
    records:
        A list of FASTA record dictionaries (as returned by :func:`parse_fasta`).
    record_id:
        The sequence identifier to locate (e.g. ``'seq1'``).

    Returns
    -------
    str
        The cleaned DNA sequence of the matching record in uppercase.

    Raises
    ------
    KeyError
        If no record with the given *record_id* exists in *records*.
    ValueError
        If the matching record does not contain a valid sequence field.

    Examples
    --------
    >>> records = [{"record_id": "seq1", "description": "seq1", "sequence": "ATGC"}]
    >>> get_sequence_by_id(records, "seq1")
    'ATGC'
    """
    for record in records:
        if isinstance(record, dict) and record.get("record_id") == record_id:
            if "sequence" not in record:
                raise ValueError(
                    f"Record with ID '{record_id}' is missing required 'sequence' field."
                )
            seq = record["sequence"]
            return clean_dna_sequence(seq)

    raise KeyError(f"Record with ID '{record_id}' not found.")


def chunk_sequence(
    sequence: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Divide a DNA sequence into chunks of the specified size.

    Parameters
    ----------
    sequence:
        The cleaned DNA sequence string to partition.
    chunk_size:
        The maximum length of each chunk. Must be a positive integer.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries representing each chunk:
        - ``chunk_index`` (int): Zero-based index of the chunk.
        - ``start``       (int): Inclusive 0-based start index.
        - ``end``         (int): Exclusive 0-based end index.
        - ``sequence``    (str): The nucleotide subsequence.
        Returns an empty list if *sequence* is empty.

    Raises
    ------
    ValueError
        If *chunk_size* is <= 0, or if *sequence* contains invalid DNA characters.

    Examples
    --------
    >>> chunk_sequence("ATGCAT", chunk_size=4)
    [{'chunk_index': 0, 'start': 0, 'end': 4, 'sequence': 'ATGC'}, {'chunk_index': 1, 'start': 4, 'end': 6, 'sequence': 'AT'}]
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size}."
        )

    if not isinstance(sequence, str):
        raise ValueError(
            f"sequence must be a string, got {type(sequence).__name__}."
        )

    if not sequence:
        return []

    validate_dna_sequence(sequence)

    chunks: list[dict[str, Any]] = []
    seq_len = len(sequence)
    chunk_index = 0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "sequence": sequence[start:end],
            }
        )
        chunk_index += 1

    return chunks


def chunk_fasta_records(
    records: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Chunk the sequences of multiple FASTA records.

    Parameters
    ----------
    records:
        A list of FASTA record dictionaries (as returned by :func:`parse_fasta`).
        Each dictionary must contain ``'record_id'`` and ``'sequence'``.
    chunk_size:
        The maximum number of nucleotides per chunk. Must be positive.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries preserving record order:
        ``[{"record_id": str, "chunks": list[dict]}]``.

    Raises
    ------
    ValueError
        If *chunk_size* is <= 0, or if any record is missing ``'record_id'``
        or ``'sequence'``, or contains an invalid sequence.

    Examples
    --------
    >>> recs = [{"record_id": "chr1", "sequence": "ATGC"}]
    >>> chunk_fasta_records(recs, chunk_size=2)
    [{'record_id': 'chr1', 'chunks': [{'chunk_index': 0, 'start': 0, 'end': 2, 'sequence': 'AT'}, {'chunk_index': 1, 'start': 2, 'end': 4, 'sequence': 'GC'}]}]
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size}."
        )

    chunked_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                f"Record must be a dictionary, got {type(record).__name__}."
            )
        if "record_id" not in record or record["record_id"] is None:
            raise ValueError(
                f"Record is missing required 'record_id' field: {record}"
            )
        if "sequence" not in record or record["sequence"] is None:
            raise ValueError(
                f"Record is missing required 'sequence' field: {record}"
            )

        seq = record["sequence"]
        if not isinstance(seq, str):
            raise ValueError(
                f"Record 'sequence' must be a string, got {type(seq).__name__}."
            )

        chunks = chunk_sequence(seq, chunk_size=chunk_size)
        chunked_records.append(
            {
                "record_id": record["record_id"],
                "chunks": chunks,
            }
        )

    return chunked_records
