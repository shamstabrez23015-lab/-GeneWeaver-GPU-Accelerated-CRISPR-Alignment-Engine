"""Alignment reporting and aggregation module for GeneWeaver."""

try:
    from src.models import AlignmentMatch
except (ImportError, ModuleNotFoundError):
    from models import AlignmentMatch


def _validate_match_object(match: object, record_id: str) -> None:
    """Validate that a match object conforms to the expected AlignmentMatch structure."""
    if isinstance(match, AlignmentMatch):
        return

    required_attrs = (
        "start",
        "end",
        "matched_sequence",
        "mismatch_count",
        "mismatch_positions",
    )
    if not all(hasattr(match, attr) for attr in required_attrs):
        raise ValueError(
            f"Malformed match object in record '{record_id}': expected AlignmentMatch or "
            f"object with attributes {required_attrs}, got {type(match).__name__}."
        )


def _validate_record(record: dict) -> None:
    """Validate that a record dictionary contains required keys."""
    if not isinstance(record, dict):
        raise ValueError(
            f"Invalid record format: expected dict, got {type(record).__name__}."
        )
    if "record_id" not in record or record["record_id"] is None:
        raise ValueError("Record is missing required 'record_id' field.")
    if "matches" not in record or record["matches"] is None:
        raise ValueError(
            f"Record '{record.get('record_id')}' is missing required 'matches' field."
        )
    if not isinstance(record["matches"], (list, tuple)):
        raise ValueError(
            f"Record '{record.get('record_id')}' matches must be a list or tuple, "
            f"got {type(record['matches']).__name__}."
        )


def flatten_alignment_results(results: list[dict]) -> list[dict]:
    """Convert nested alignment results into a flat list suitable for tabular display.

    Each dictionary in the returned list represents a single alignment match and
    contains the fields:
        - record_id
        - start
        - end
        - matched_sequence
        - mismatch_count
        - mismatch_positions

    The input list and nested objects are not modified. Records with zero matches
    are skipped in the flattened match list without error.

    Args:
        results: A list of record dictionaries with 'record_id' and 'matches'.

    Returns:
        A flat list of dictionaries containing individual match details with record_id.

    Raises:
        ValueError: If results is not a list, if a record lacks 'record_id' or 'matches',
                    or if any match object is malformed.
    """
    if not isinstance(results, list):
        raise ValueError(f"results must be a list of dicts, got {type(results).__name__}.")

    flattened: list[dict] = []

    for record in results:
        _validate_record(record)
        record_id = record["record_id"]

        for match in record["matches"]:
            _validate_match_object(match, record_id)
            flattened.append({
                "record_id": record_id,
                "start": match.start,
                "end": match.end,
                "matched_sequence": match.matched_sequence,
                "mismatch_count": match.mismatch_count,
                "mismatch_positions": list(match.mismatch_positions),
            })

    return flattened


def summarize_alignment_results(results: list[dict]) -> dict:
    """Calculate summary statistics from nested alignment results.

    Computes:
        - total_records: Total number of FASTA records processed.
        - records_with_matches: Number of records with at least one match.
        - total_matches: Total number of matches across all records.
        - total_mismatches: Cumulative number of mismatches across all matches.
        - average_mismatches_per_match: Average mismatches per match (0.0 if no matches).

    Args:
        results: A list of record dictionaries with 'record_id' and 'matches'.

    Returns:
        A dictionary containing summary statistics.

    Raises:
        ValueError: If results is not a list, if a record is missing required fields,
                    or if any match object is malformed.
    """
    if not isinstance(results, list):
        raise ValueError(f"results must be a list of dicts, got {type(results).__name__}.")

    total_records = len(results)
    records_with_matches = 0
    total_matches = 0
    total_mismatches = 0

    for record in results:
        _validate_record(record)
        matches = record["matches"]
        record_id = record["record_id"]

        if len(matches) > 0:
            records_with_matches += 1

        for match in matches:
            _validate_match_object(match, record_id)
            total_matches += 1
            total_mismatches += match.mismatch_count

    avg_mismatches = (
        float(total_mismatches) / total_matches if total_matches > 0 else 0.0
    )

    return {
        "total_records": total_records,
        "records_with_matches": records_with_matches,
        "total_matches": total_matches,
        "total_mismatches": total_mismatches,
        "average_mismatches_per_match": avg_mismatches,
    }

