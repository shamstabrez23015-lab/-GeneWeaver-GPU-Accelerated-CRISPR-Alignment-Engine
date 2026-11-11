"""
genomic_parser.py
-----------------
Day 1 – GeneWeaver: Genomic Parser Module

Provides FASTA file parsing utilities using BioPython's SeqIO module.
Responsible for reading genomic FASTA records and returning structured data
ready for downstream processing (alignment, validation, etc.).

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

