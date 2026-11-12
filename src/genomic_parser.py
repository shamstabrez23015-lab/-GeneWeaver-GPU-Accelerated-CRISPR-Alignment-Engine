"""
genomic_parser.py
-----------------
Day 1 & Day 2 – GeneWeaver: Genomic Parser Module

Provides FASTA file parsing and sequence extraction utilities using BioPython's SeqIO module.
Responsible for reading genomic FASTA records, extracting DNA sequences, and
retrieving sequences by record ID ready for downstream processing (alignment, validation, etc.).

Note: Sequence cleaning, validation, and genome chunking are deferred to
later days and are intentionally *not* implemented here.
"""

from __future__ import annotations

import os
from typing import Any

from Bio import SeqIO


def parse_fasta(file_path: str) -> list[dict[str, Any]]:
    """Parse a FASTA file and return its records as a list of dictionaries.

    Each record is represented as a dictionary with the following keys:

    - ``record_id``   (str): The sequence identifier (text before the first
                             space on the header line, e.g. ``seq1``).
    - ``description`` (str): The full header string, including the identifier
                             (e.g. ``seq1 Human BRCA1 exon 2``).
    - ``sequence``    (str): The DNA sequence in uppercase (multi-line FASTA
                             sequences are automatically concatenated).

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
        If the FASTA file exists but contains no sequence records.

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
            records.append(
                {
                    "record_id": seq_record.id,
                    "description": seq_record.description,
                    "sequence": str(seq_record.seq).upper(),
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
    sequences are uppercase strings.

    Parameters
    ----------
    records:
        A list of record dictionaries, such as those returned by :func:`parse_fasta`.
        Each dictionary must contain a valid ``'sequence'`` key with a non-empty string.

    Returns
    -------
    list[str]
        A list of uppercase DNA sequence strings in the same order as *records*.
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
        if not isinstance(seq, str) or not seq.strip():
            raise ValueError(
                f"Record contains an invalid or empty sequence field: {record}"
            )
        sequences.append(seq.upper())

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
        The DNA sequence of the matching record in uppercase.

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
            if not isinstance(seq, str) or not seq.strip():
                raise ValueError(
                    f"Record with ID '{record_id}' contains an invalid or empty sequence field."
                )
            return seq.upper()

    raise KeyError(f"Record with ID '{record_id}' not found.")
