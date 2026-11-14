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

Day 3:
15. Lowercase DNA conversion in sequence cleaning.
16. Removing spaces, tabs, and newline characters.
17. Allowing valid A, T, G, C, and N bases.
18. Rejecting unsupported characters such as X or Z.
19. Rejecting empty or whitespace-only sequences.
20. Validating correct sequences (returns True).
21. Validating incorrect sequences (raises ValueError).
22. Parsing a FASTA file with lowercase sequences.
23. Rejecting an invalid sequence in a FASTA record.

Day 4:
24. Splitting a sequence into equal-sized chunks.
25. Handling a smaller final chunk.
26. Correct zero-based chunk indexes.
27. Correct start (inclusive) and end (exclusive) coordinates.
28. Reconstructing the original sequence by joining chunks.
29. Rejecting zero and negative chunk size.
30. Handling an empty sequence (returns empty list).
31. Chunking multiple FASTA records.
32. Preserving record IDs and record order.
33. Handling records with empty sequences (includes record with empty chunks).
34. Validating missing record ID and sequence fields.
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
    clean_dna_sequence,
    chunk_fasta_records,
    chunk_sequence,
    extract_sequences,
    get_sequence_by_id,
    parse_fasta,
    validate_dna_sequence,
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
# Day 2 Tests: extract_sequences (Preserved)
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
        valid_seqs = ["A", "C", "G", "T", "N", "AC", "GT", "TG", "CA", "ATGC"]
        mock_records = [
            {"record_id": f"id_{i}", "sequence": valid_seqs[i]}
            for i in range(10)
        ]
        expected = valid_seqs[:]
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
        with pytest.raises(ValueError, match="string|sequence"):
            extract_sequences(invalid_records)

    def test_empty_string_sequence_raises_value_error(self):
        """A record with an empty sequence string must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "sequence": ""}
        ]
        with pytest.raises(ValueError, match="empty"):
            extract_sequences(invalid_records)

    def test_non_string_sequence_field_raises_value_error(self):
        """A record with a non-string sequence must raise ValueError."""
        invalid_records = [
            {"record_id": "s1", "sequence": 12345}
        ]
        with pytest.raises(ValueError, match="string"):
            extract_sequences(invalid_records)

    def test_non_dict_record_raises_value_error(self):
        """Passing non-dictionary items in the records list must raise ValueError."""
        invalid_records = ["not a dict"]  # type: ignore[list-item]
        with pytest.raises(ValueError):
            extract_sequences(invalid_records)


# ---------------------------------------------------------------------------
# Day 2 Tests: get_sequence_by_id (Preserved)
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
        with pytest.raises(ValueError, match="empty"):
            get_sequence_by_id(mock_records, "bad_record")


# ---------------------------------------------------------------------------
# Day 2 Tests: Record Format Preservation & Non-Mutation (Preserved)
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


# ---------------------------------------------------------------------------
# Day 3 Tests: clean_dna_sequence and validate_dna_sequence (Preserved)
# ---------------------------------------------------------------------------


class TestCleanDnaSequence:
    """Tests for clean_dna_sequence(sequence)."""

    def test_lowercase_dna_conversion(self):
        """Lowercase nucleotides must be converted to uppercase."""
        assert clean_dna_sequence("atgc") == "ATGC"
        assert clean_dna_sequence("atgcn") == "ATGCN"
        assert clean_dna_sequence("aTgCn") == "ATGCN"

    def test_removing_spaces_and_newline_characters(self):
        """Spaces, tabs, carriage returns, and newlines must be stripped from sequence."""
        assert clean_dna_sequence(" A T G C ") == "ATGC"
        assert clean_dna_sequence("A\tT\nG\r\nC\tN") == "ATGCN"
        assert clean_dna_sequence("  ATGC  \n  ACTG  \r\n  GGTT  ") == "ATGCACTGGGTT"

    def test_valid_bases_atgcn(self):
        """A, T, G, C, and N are all permitted and preserved."""
        assert clean_dna_sequence("A") == "A"
        assert clean_dna_sequence("T") == "T"
        assert clean_dna_sequence("G") == "G"
        assert clean_dna_sequence("C") == "C"
        assert clean_dna_sequence("N") == "N"
        assert clean_dna_sequence("ATGCN") == "ATGCN"
        assert clean_dna_sequence("NNNN") == "NNNN"

    def test_rejecting_unsupported_characters_x_or_z(self):
        """Unsupported characters such as X or Z must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported DNA character"):
            clean_dna_sequence("ATGCX")
        with pytest.raises(ValueError, match="Unsupported DNA character"):
            clean_dna_sequence("ZATG")
        with pytest.raises(ValueError, match="Unsupported DNA character"):
            clean_dna_sequence("ATGCXZ")
        with pytest.raises(ValueError, match="Unsupported DNA character"):
            clean_dna_sequence("ATGC-NN")

    def test_rejecting_empty_sequences(self):
        """Empty sequence or whitespace-only sequence must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            clean_dna_sequence("")
        with pytest.raises(ValueError, match="empty"):
            clean_dna_sequence("   ")
        with pytest.raises(ValueError, match="empty"):
            clean_dna_sequence("\t\n\r  ")

    def test_non_string_sequence_raises_value_error(self):
        """Non-string sequence input must raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            clean_dna_sequence(12345)  # type: ignore[arg-type]


class TestValidateDnaSequence:
    """Tests for validate_dna_sequence(sequence)."""

    def test_validating_a_correct_sequence(self):
        """Valid non-empty sequences should return True."""
        assert validate_dna_sequence("ATGC") is True
        assert validate_dna_sequence("atgcn") is True
        assert validate_dna_sequence("  A T \t G \n C N \r\n") is True
        assert validate_dna_sequence("NNNN") is True

    def test_validating_an_incorrect_sequence(self):
        """Invalid sequences must raise ValueError."""
        with pytest.raises(ValueError):
            validate_dna_sequence("ATGCX")
        with pytest.raises(ValueError):
            validate_dna_sequence("ZATGC")
        with pytest.raises(ValueError):
            validate_dna_sequence("")
        with pytest.raises(ValueError):
            validate_dna_sequence("   \n\t")


class TestParseFastaCleaningAndValidation:
    """Tests for parse_fasta sequence cleaning and validation."""

    def test_parsing_fasta_with_lowercase_sequences(self, tmp_path):
        """FASTA files with lowercase sequences must be cleaned and uppercased."""
        content = (
            ">seq1 BRCA1 fragment\natgcttag\n"
            ">seq2 TP53 fragment\nggccnntt\n"
        )
        path = _write_fasta(tmp_path, "lower_valid.fasta", content)
        records = parse_fasta(path)
        assert len(records) == 2
        assert records[0]["record_id"] == "seq1"
        assert records[0]["description"] == "seq1 BRCA1 fragment"
        assert records[0]["sequence"] == "ATGCTTAG"
        assert records[1]["record_id"] == "seq2"
        assert records[1]["description"] == "seq2 TP53 fragment"
        assert records[1]["sequence"] == "GGCCNNTT"

    def test_parsing_fasta_with_spaces_and_newlines_in_sequence(self, tmp_path):
        """FASTA records with internal whitespace or multiline breaks are cleaned."""
        content = (
            ">seq1 Multiline with spaces\nAT GC\nTT GG\nNN CC\n"
        )
        path = _write_fasta(tmp_path, "whitespace.fasta", content)
        records = parse_fasta(path)
        assert records[0]["sequence"] == "ATGCTTGGNNCC"

    def test_rejecting_invalid_sequence_in_fasta_record(self, tmp_path):
        """FASTA records with invalid characters (e.g. X, Z) must raise ValueError."""
        content = (
            ">valid_record\nATGC\n"
            ">corrupt_record\nATGCXNN\n"
        )
        path = _write_fasta(tmp_path, "corrupt.fasta", content)
        with pytest.raises(ValueError, match="corrupt_record|Unsupported"):
            parse_fasta(path)

    def test_rejecting_empty_sequence_in_fasta_record(self, tmp_path):
        """FASTA records with empty sequence must raise ValueError."""
        content = (
            ">empty_seq_record\n\n"
        )
        path = _write_fasta(tmp_path, "empty_seq.fasta", content)
        with pytest.raises(ValueError):
            parse_fasta(path)


# ---------------------------------------------------------------------------
# Day 4 Tests: chunk_sequence and chunk_fasta_records
# ---------------------------------------------------------------------------


class TestChunkSequence:
    """Tests for chunk_sequence(sequence, chunk_size)."""

    def test_splitting_sequence_into_equal_chunks(self):
        """A sequence divided evenly into equal-sized chunks."""
        seq = "ATGC" * 15  # 60 bp
        chunk_size = 20
        chunks = chunk_sequence(seq, chunk_size=chunk_size)

        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk["sequence"]) == 20

    def test_handling_smaller_final_chunk(self):
        """The final chunk can be smaller than chunk_size."""
        seq = "ACGTACGTNN"  # 10 bp
        chunk_size = 4
        chunks = chunk_sequence(seq, chunk_size=chunk_size)

        assert len(chunks) == 3
        assert len(chunks[0]["sequence"]) == 4
        assert len(chunks[1]["sequence"]) == 4
        assert len(chunks[2]["sequence"]) == 2
        assert chunks[2]["sequence"] == "NN"

    def test_correct_chunk_indexes(self):
        """Chunk indexes must be zero-based sequential integers."""
        seq = "ATGC" * 10  # 40 bp
        chunks = chunk_sequence(seq, chunk_size=10)

        assert len(chunks) == 4
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]

    def test_correct_start_and_end_coordinates(self):
        """Start coordinate must be inclusive and end coordinate exclusive."""
        seq = "ACGTACGTNN"  # 10 bp
        chunks = chunk_sequence(seq, chunk_size=4)

        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 4
        assert chunks[1]["start"] == 4
        assert chunks[1]["end"] == 8
        assert chunks[2]["start"] == 8
        assert chunks[2]["end"] == 10

        for chunk in chunks:
            assert seq[chunk["start"]:chunk["end"]] == chunk["sequence"]

    def test_reconstructing_original_sequence_by_joining_chunks(self):
        """Joining all chunk sequences must reconstruct the original sequence exactly."""
        seq = "ACGT" * 25 + "NNNN"  # 104 bp
        chunks = chunk_sequence(seq, chunk_size=15)
        reconstructed = "".join(chunk["sequence"] for chunk in chunks)
        assert reconstructed == seq

    def test_rejecting_zero_chunk_size(self):
        """chunk_size=0 must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            chunk_sequence("ATGC", chunk_size=0)

    def test_rejecting_negative_chunk_size(self):
        """chunk_size < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            chunk_sequence("ATGC", chunk_size=-10)

    def test_handling_empty_sequence(self):
        """An empty sequence must return an empty list of chunks."""
        assert chunk_sequence("", chunk_size=100) == []

    def test_chunk_sequence_validates_sequence(self):
        """Sequences with unsupported characters must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported DNA character"):
            chunk_sequence("ATGCXNN", chunk_size=10)

    def test_chunk_sequence_default_chunk_size(self):
        """Calling chunk_sequence without chunk_size uses DEFAULT_CHUNK_SIZE."""
        seq = "ATGC" * 10
        chunks = chunk_sequence(seq)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 40
        assert chunks[0]["sequence"] == seq


class TestChunkFastaRecords:
    """Tests for chunk_fasta_records(records, chunk_size)."""

    def test_chunking_multiple_fasta_records_from_file(self, tmp_path):
        """Chunk multiple records parsed from a FASTA file."""
        content = (
            ">chr1 Chromosome 1\n"
            "ACGTACGTACGT\n"  # 12 bp
            ">chr2 Chromosome 2\n"
            "GGCCAATT\n"        # 8 bp
        )
        path = _write_fasta(tmp_path, "multi_chunk.fasta", content)
        records = parse_fasta(path)
        chunked = chunk_fasta_records(records, chunk_size=5)

        assert len(chunked) == 2
        assert chunked[0]["record_id"] == "chr1"
        assert len(chunked[0]["chunks"]) == 3  # 5, 5, 2
        assert chunked[0]["chunks"][0]["sequence"] == "ACGTA"
        assert chunked[0]["chunks"][1]["sequence"] == "CGTAC"
        assert chunked[0]["chunks"][2]["sequence"] == "GT"

        assert chunked[1]["record_id"] == "chr2"
        assert len(chunked[1]["chunks"]) == 2  # 5, 3
        assert chunked[1]["chunks"][0]["sequence"] == "GGCCA"
        assert chunked[1]["chunks"][1]["sequence"] == "ATT"

    def test_preserving_record_ids_and_record_order(self):
        """Record IDs and original ordering must be preserved."""
        mock_records = [
            {"record_id": f"scaffold_{i}", "sequence": "ACGT" * 5}
            for i in range(5)
        ]
        chunked = chunk_fasta_records(mock_records, chunk_size=10)
        assert [r["record_id"] for r in chunked] == [f"scaffold_{i}" for i in range(5)]

    def test_handling_record_with_empty_sequence(self):
        """Records with empty sequences should be included with empty chunk lists."""
        mock_records = [
            {"record_id": "valid_rec", "sequence": "ATGC"},
            {"record_id": "empty_rec", "sequence": ""},
        ]
        chunked = chunk_fasta_records(mock_records, chunk_size=2)
        assert len(chunked) == 2
        assert chunked[0]["record_id"] == "valid_rec"
        assert len(chunked[0]["chunks"]) == 2
        assert chunked[1]["record_id"] == "empty_rec"
        assert chunked[1]["chunks"] == []

    def test_handling_missing_sequence_field(self):
        """Record missing 'sequence' key must raise ValueError."""
        bad_records = [
            {"record_id": "chr1"}
        ]
        with pytest.raises(ValueError, match="sequence"):
            chunk_fasta_records(bad_records, chunk_size=10)

    def test_handling_none_sequence_field(self):
        """Record with sequence=None must raise ValueError."""
        bad_records = [
            {"record_id": "chr1", "sequence": None}
        ]
        with pytest.raises(ValueError, match="sequence"):
            chunk_fasta_records(bad_records, chunk_size=10)

    def test_handling_missing_record_id_field(self):
        """Record missing 'record_id' must raise ValueError."""
        bad_records = [
            {"sequence": "ATGC"}
        ]
        with pytest.raises(ValueError, match="record_id"):
            chunk_fasta_records(bad_records, chunk_size=10)

    def test_rejecting_non_dict_record(self):
        """Passing non-dict records must raise ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            chunk_fasta_records(["not a dict"], chunk_size=10)  # type: ignore[list-item]

    def test_rejecting_zero_or_negative_chunk_size(self):
        """Non-positive chunk size must raise ValueError."""
        mock_records = [{"record_id": "chr1", "sequence": "ATGC"}]
        with pytest.raises(ValueError, match="positive"):
            chunk_fasta_records(mock_records, chunk_size=0)
        with pytest.raises(ValueError, match="positive"):
            chunk_fasta_records(mock_records, chunk_size=-5)

    def test_coordinates_are_relative_to_original_record(self):
        """Coordinates start and end are 0-based relative to each original record."""
        mock_records = [
            {"record_id": "chrA", "sequence": "AAAAATTTTT"},
            {"record_id": "chrB", "sequence": "CCCCCGGGGG"},
        ]
        chunked = chunk_fasta_records(mock_records, chunk_size=5)

        # Both records must have their first chunk start at 0
        assert chunked[0]["chunks"][0]["start"] == 0
        assert chunked[0]["chunks"][0]["end"] == 5
        assert chunked[0]["chunks"][1]["start"] == 5
        assert chunked[0]["chunks"][1]["end"] == 10

        assert chunked[1]["chunks"][0]["start"] == 0
        assert chunked[1]["chunks"][0]["end"] == 5
        assert chunked[1]["chunks"][1]["start"] == 5
        assert chunked[1]["chunks"][1]["end"] == 10
