"""
test_genomic_parser.py
----------------------
Pytest test suite for src/genomic_parser.py

Tests are self-contained and use pytest's ``tmp_path`` fixture so that no
permanent files are created or left behind.

Covered scenarios
-----------------
1. Parsing a single FASTA record.
2. Parsing multiple FASTA records.
3. Lowercase sequences are returned in uppercase.
4. record_id and description are extracted correctly.
5. Missing file raises FileNotFoundError.
6. Empty FASTA file raises ValueError.
"""

from __future__ import annotations

import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Make the sibling ``src`` package importable when running pytest from the
# project root or from the ``tests`` directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from genomic_parser import parse_fasta  # noqa: E402  (import after sys.path tweak)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path, filename: str, content: str):
    """Write *content* to a file inside *tmp_path* and return its path."""
    fasta_file = tmp_path / filename
    fasta_file.write_text(content, encoding="utf-8")
    return str(fasta_file)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleRecord:
    """parse_fasta correctly reads one FASTA record."""

    def test_returns_a_list(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "single.fasta",
            ">seq1 Human BRCA1 exon 1\nATGCATGC\n",
        )
        result = parse_fasta(path)
        assert isinstance(result, list)

    def test_list_has_one_element(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "single.fasta",
            ">seq1 Human BRCA1 exon 1\nATGCATGC\n",
        )
        result = parse_fasta(path)
        assert len(result) == 1

    def test_contains_required_keys(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "single.fasta",
            ">seq1 Human BRCA1 exon 1\nATGCATGC\n",
        )
        record = parse_fasta(path)[0]
        assert "record_id" in record
        assert "description" in record
        assert "sequence" in record

    def test_sequence_value(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "single.fasta",
            ">seq1 Human BRCA1 exon 1\nATGCATGC\n",
        )
        record = parse_fasta(path)[0]
        assert record["sequence"] == "ATGCATGC"

    def test_multiline_sequence_is_concatenated(self, tmp_path):
        """Multi-line sequences should be joined into one continuous string."""
        path = _write_fasta(
            tmp_path,
            "multiline.fasta",
            ">seq1 Exon fragment\nATGC\nACTG\nGGTT\n",
        )
        record = parse_fasta(path)[0]
        assert record["sequence"] == "ATGCACTGGGTT"


class TestMultipleRecords:
    """parse_fasta correctly reads more than one FASTA record."""

    CONTENT = (
        ">alpha Gene alpha coding sequence\nAAAAAAAA\n"
        ">beta Gene beta coding sequence\nTTTTTTTT\n"
        ">gamma Gene gamma coding sequence\nCCCCCCCC\n"
    )

    def test_returns_correct_count(self, tmp_path):
        path = _write_fasta(tmp_path, "multi.fasta", self.CONTENT)
        result = parse_fasta(path)
        assert len(result) == 3

    def test_all_records_have_required_keys(self, tmp_path):
        path = _write_fasta(tmp_path, "multi.fasta", self.CONTENT)
        for record in parse_fasta(path):
            assert "record_id" in record
            assert "description" in record
            assert "sequence" in record

    def test_sequences_are_in_order(self, tmp_path):
        path = _write_fasta(tmp_path, "multi.fasta", self.CONTENT)
        result = parse_fasta(path)
        assert result[0]["sequence"] == "AAAAAAAA"
        assert result[1]["sequence"] == "TTTTTTTT"
        assert result[2]["sequence"] == "CCCCCCCC"


class TestUppercaseConversion:
    """Lowercase nucleotides are returned as uppercase."""

    def test_all_lowercase_converted(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "lower.fasta",
            ">seq1 lowercase test\natgcttag\n",
        )
        record = parse_fasta(path)[0]
        assert record["sequence"] == "ATGCTTAG"

    def test_mixed_case_converted(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "mixed.fasta",
            ">seq1 mixed case test\nAtGcTtAg\n",
        )
        record = parse_fasta(path)[0]
        assert record["sequence"] == "ATGCTTAG"

    def test_already_uppercase_unchanged(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "upper.fasta",
            ">seq1 uppercase test\nATGCTTAG\n",
        )
        record = parse_fasta(path)[0]
        assert record["sequence"] == "ATGCTTAG"

    def test_multiple_records_all_uppercase(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "multi_case.fasta",
            ">r1 record 1\natgc\n>r2 record 2\nTTgg\n",
        )
        result = parse_fasta(path)
        assert result[0]["sequence"] == "ATGC"
        assert result[1]["sequence"] == "TTGG"


class TestRecordIdAndDescription:
    """record_id and description are extracted correctly."""

    def test_record_id_is_extracted(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "id_desc.fasta",
            ">NM_007294.3 Homo sapiens BRCA1 mRNA\nATGC\n",
        )
        record = parse_fasta(path)[0]
        assert record["record_id"] == "NM_007294.3"

    def test_description_contains_full_header(self, tmp_path):
        """BioPython includes the ID in the description field."""
        path = _write_fasta(
            tmp_path,
            "id_desc.fasta",
            ">NM_007294.3 Homo sapiens BRCA1 mRNA\nATGC\n",
        )
        record = parse_fasta(path)[0]
        assert record["description"] == "NM_007294.3 Homo sapiens BRCA1 mRNA"

    def test_id_only_header_no_description_tokens(self, tmp_path):
        """When the header has no space, ID and description are both the ID."""
        path = _write_fasta(
            tmp_path,
            "id_only.fasta",
            ">seq1\nATGC\n",
        )
        record = parse_fasta(path)[0]
        assert record["record_id"] == "seq1"
        # BioPython sets description == id when there is no additional text
        assert record["description"] == "seq1"

    def test_multiple_records_ids_in_order(self, tmp_path):
        path = _write_fasta(
            tmp_path,
            "multi_id.fasta",
            ">first First record\nAAAA\n>second Second record\nTTTT\n",
        )
        result = parse_fasta(path)
        assert result[0]["record_id"] == "first"
        assert result[1]["record_id"] == "second"


class TestMissingFile:
    """parse_fasta raises FileNotFoundError for non-existent paths."""

    def test_nonexistent_path_raises(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.fasta")
        with pytest.raises(FileNotFoundError):
            parse_fasta(missing)

    def test_error_message_contains_path(self, tmp_path):
        missing = str(tmp_path / "ghost.fasta")
        with pytest.raises(FileNotFoundError, match="ghost.fasta"):
            parse_fasta(missing)

    def test_directory_path_raises(self, tmp_path):
        """Passing a directory path (not a file) should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_fasta(str(tmp_path))


class TestEmptyFastaFile:
    """parse_fasta raises ValueError for a FASTA file with no records."""

    def test_empty_file_raises_value_error(self, tmp_path):
        path = _write_fasta(tmp_path, "empty.fasta", "")
        with pytest.raises(ValueError):
            parse_fasta(path)

    def test_error_message_mentions_no_records(self, tmp_path):
        path = _write_fasta(tmp_path, "empty.fasta", "")
        with pytest.raises(ValueError, match="No FASTA records"):
            parse_fasta(path)

