"""
test_genomic_parser.py
----------------------
Pytest test suite for src/genomic_parser.py

Tests are self-contained and use pytest's ``tmp_path`` fixture so that no
permanent files are created or left behind.

Covered scenarios
-----------------
Day 1:
1. Parsing a single FASTA record.
2. Parsing multiple FASTA records.
3. Lowercase sequences are returned in uppercase.
4. record_id and description are extracted correctly.
5. Missing file raises FileNotFoundError.
6. Empty FASTA file raises ValueError.

Day 2:
7. Extracting sequences from multiple FASTA records.
8. Preserving record order.
9. Extracting a single record by ID.
10. Handling a missing record ID.
11. Handling an empty records list.
12. Handling a record with a missing or invalid sequence field.
13. Ensuring extracted sequences are uppercase.
14. Preserving the existing record format without mutation.
"""

from __future__ import annotations

import copy
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

from genomic_parser import (  # noqa: E402  (import after sys.path tweak)
    extract_sequences,
    get_sequence_by_id,
    parse_fasta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path, filename: str, content: str):
    """Write *content* to a file inside *tmp_path* and return its path."""
    fasta_file = tmp_path / filename
    fasta_file.write_text(content, encoding="utf-8")
    return str(fasta_file)


# ---------------------------------------------------------------------------
# Day 1 Tests (Preserved)
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


# ---------------------------------------------------------------------------
# Day 2 Tests: extract_sequences
# ---------------------------------------------------------------------------


class TestExtractSequences:
    """Tests for extract_sequences(records)."""

    def test_extract_from_multiple_fasta_records(self, tmp_path):
        """Extract sequences from records parsed from a multi-record FASTA file."""
        content = (
            ">g1 Gene 1\nACGTACGT\n"
            ">g2 Gene 2\nTGCATGCA\n"
            ">g3 Gene 3\nGGCCAAAT\n"
        )
        path = _write_fasta(tmp_path, "multi_extract.fasta", content)
        records = parse_fasta(path)
        sequences = extract_sequences(records)
        assert sequences == ["ACGTACGT", "TGCATGCA", "GGCCAAAT"]

    def test_extract_from_mock_records(self):
        """Extract sequences from mock record dictionaries."""
        mock_records = [
            {"record_id": "r1", "description": "desc 1", "sequence": "ATGC"},
            {"record_id": "r2", "description": "desc 2", "sequence": "CGTA"},
            {"record_id": "r3", "description": "desc 3", "sequence": "TTTT"},
        ]
        sequences = extract_sequences(mock_records)
        assert sequences == ["ATGC", "CGTA", "TTTT"]

    def test_preserves_record_order(self):
        """Original record ordering must be strictly preserved in output."""
        mock_records = [
            {"record_id": f"id_{i}", "sequence": f"SEQ{i}"}
            for i in range(10)
        ]
        expected = [f"SEQ{i}" for i in range(10)]
        assert extract_sequences(mock_records) == expected

    def test_empty_records_list_returns_empty_list(self):
        """An empty input list should return an empty list."""
        assert extract_sequences([]) == []

    def test_ensures_extracted_sequences_are_uppercase(self):
        """Sequences should be converted to uppercase."""
        mock_records = [
            {"record_id": "s1", "sequence": "atgc"},
            {"record_id": "s2", "sequence": "AtGcN"},
            {"record_id": "s3", "sequence": "cgtan"},
        ]
        sequences = extract_sequences(mock_records)
        assert sequences == ["ATGC", "ATGCN", "CGTAN"]

    def test_missing_sequence_field_raises_value_error(self):
        """A record missing the 'sequence' key must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "description": "missing sequence"}
        ]
        with pytest.raises(ValueError, match="sequence"):
            extract_sequences(invalid_records)

    def test_none_sequence_field_raises_value_error(self):
        """A record with None as the sequence value must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "sequence": None}
        ]
        with pytest.raises(ValueError, match="invalid|sequence"):
            extract_sequences(invalid_records)

    def test_empty_string_sequence_raises_value_error(self):
        """A record with an empty sequence string must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "sequence": ""}
        ]
        with pytest.raises(ValueError, match="invalid|empty"):
            extract_sequences(invalid_records)

    def test_non_string_sequence_field_raises_value_error(self):
        """A record with a non-string sequence must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "sequence": 12345}
        ]
        with pytest.raises(ValueError, match="invalid|sequence"):
            extract_sequences(invalid_records)

    def test_non_dict_record_raises_value_error(self):
        """Passing non-dictionary items in the records list must raise ValueError."""
        invalid_records = ["not a dict"]  # type: ignore[list-item]
        with pytest.raises(ValueError):
            extract_sequences(invalid_records)


# ---------------------------------------------------------------------------
# Day 2 Tests: get_sequence_by_id
# ---------------------------------------------------------------------------


class TestGetSequenceById:
    """Tests for get_sequence_by_id(records, record_id)."""

    def test_extract_single_record_by_id_from_fasta(self, tmp_path):
        """Locate and return a sequence from parsed FASTA records."""
        content = (
            ">alpha Gene alpha\nAAAAAAAA\n"
            ">beta Gene beta\nTTTTTTTT\n"
            ">gamma Gene gamma\nCCCCCCCC\n"
        )
        path = _write_fasta(tmp_path, "lookup.fasta", content)
        records = parse_fasta(path)
        seq_beta = get_sequence_by_id(records, "beta")
        assert seq_beta == "TTTTTTTT"

    def test_extract_single_record_by_id_from_mock(self):
        """Locate and return a sequence from mock records."""
        mock_records = [
            {"record_id": "chr1", "sequence": "ACGTACGT"},
            {"record_id": "chr2", "sequence": "TGCATGCA"},
        ]
        assert get_sequence_by_id(mock_records, "chr1") == "ACGTACGT"
        assert get_sequence_by_id(mock_records, "chr2") == "TGCATGCA"

    def test_missing_record_id_raises_key_error(self):
        """Querying a non-existent record ID must raise KeyError."""
        mock_records = [
            {"record_id": "chr1", "sequence": "ACGT"},
        ]
        with pytest.raises(KeyError, match="chr99"):
            get_sequence_by_id(mock_records, "chr99")

    def test_empty_records_list_raises_key_error(self):
        """Searching an empty list must raise KeyError."""
        with pytest.raises(KeyError, match="seq1"):
            get_sequence_by_id([], "seq1")

    def test_returns_uppercase_sequence(self):
        """Returned sequence must be in uppercase."""
        mock_records = [
            {"record_id": "lower_id", "sequence": "atgcn"},
        ]
        assert get_sequence_by_id(mock_records, "lower_id") == "ATGCN"

    def test_matched_record_missing_sequence_raises_value_error(self):
        """If the requested record exists but has no sequence field, raise ValueError."""
        mock_records = [
            {"record_id": "bad_record", "description": "no sequence key"}
        ]
        with pytest.raises(ValueError, match="sequence"):
            get_sequence_by_id(mock_records, "bad_record")

    def test_matched_record_empty_sequence_raises_value_error(self):
        """If the requested record has an empty sequence, raise ValueError."""
        mock_records = [
            {"record_id": "bad_record", "sequence": "   "}
        ]
        with pytest.raises(ValueError, match="sequence"):
            get_sequence_by_id(mock_records, "bad_record")


# ---------------------------------------------------------------------------
# Day 2 Tests: Record Format Preservation & Non-Mutation
# ---------------------------------------------------------------------------


class TestRecordFormatPreservation:
    """Verify that records format is preserved and input list is not mutated."""

    def test_parse_fasta_record_format_preserved(self, tmp_path):
        """Verify parse_fasta return format matches requirement."""
        path = _write_fasta(
            tmp_path,
            "format_check.fasta",
            ">seq1 description here\nACGT\n",
        )
        records = parse_fasta(path)
        assert len(records) == 1
        record = records[0]
        assert set(record.keys()) == {"record_id", "description", "sequence"}
        assert record["record_id"] == "seq1"
        assert record["description"] == "seq1 description here"
        assert record["sequence"] == "ACGT"

    def test_extract_sequences_does_not_mutate_input(self):
        """extract_sequences must not modify the records list or dicts in place."""
        original_records = [
            {"record_id": "r1", "description": "desc1", "sequence": "atgc"},
            {"record_id": "r2", "description": "desc2", "sequence": "cgta"},
        ]
        snapshot = copy.deepcopy(original_records)
        extract_sequences(original_records)
        assert original_records == snapshot

    def test_get_sequence_by_id_does_not_mutate_input(self):
        """get_sequence_by_id must not modify the records list or dicts in place."""
        original_records = [
            {"record_id": "r1", "description": "desc1", "sequence": "atgc"},
        ]
        snapshot = copy.deepcopy(original_records)
        get_sequence_by_id(original_records, "r1")
        assert original_records == snapshot
