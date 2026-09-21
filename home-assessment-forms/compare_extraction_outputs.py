#!/usr/bin/env python3
"""
Bake-off: compares two home-assessment extraction outputs field-by-field, matched by
source_file. Built to compare the Cloud Vision OCR baseline
(extract_home_assessment_forms_cloud_vision.py) against the single-pass Gemini approach
(extract_home_assessment_forms_gemini.py), run on the SAME set of forms. (Field definitions
are still imported from extract_home_assessment_forms_docai.py -- that module stays as the
shared question-text/field-key source of truth even though Document AI itself is no longer
part of this pipeline.)

Run both extractors against the same --limit N pilot batch first, then run this to see:
  - overall field-level agreement rate between the two
  - which specific fields disagree most, with both values shown side by side for spot-checking
  - Daily Function agreement specifically (the section neither approach resolves with full
    confidence on its own)
  - how many fields each approach itself flagged as low-confidence/uncertain -- a rough proxy
    for how much manual review each path would leave behind at 100-form scale

This doesn't tell you which one is "right" by itself -- a mismatch means the two disagree, not
that the baseline is correct. Use the mismatch CSV to spot-check a sample of disagreements
against the actual scans before deciding which approach (or which fields from which approach)
to trust at full scale.

Usage:
    python compare_extraction_outputs.py \\
        --baseline ./home_assessment_extract_cloud_vision.json \\
        --candidate ./home_assessment_extract_gemini.json \\
        --out-prefix ./comparison_cloud_vision_vs_gemini
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from extract_home_assessment_forms_docai import QUESTION_TEXT, DAILY_FUNCTION_LABELS, flatten_for_csv

ALL_FIELD_KEYS = list(QUESTION_TEXT.keys()) + list(DAILY_FUNCTION_LABELS.keys())
DAILY_FUNCTION_KEYS = set(DAILY_FUNCTION_LABELS.keys())


def normalize(value):
    """Loose equality for comparison purposes: None/empty-string/whitespace-only all count as
    "no answer" so a baseline that stores None and a candidate that stores "" don't show up as
    a false mismatch; strings are compared case- and whitespace-insensitively."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return None if text == "" else text


def load_records(path: str) -> dict:
    records = json.loads(Path(path).read_text())
    return {r["source_file"]: r for r in records}


def get_nested(record: dict, dotted_key: str):
    cur = record
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="Cloud Vision OCR output JSON")
    ap.add_argument("--candidate", required=True, help="Gemini output JSON")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    baseline = load_records(args.baseline)
    candidate = load_records(args.candidate)

    shared_files = sorted(set(baseline) & set(candidate))
    only_baseline = sorted(set(baseline) - set(candidate))
    only_candidate = sorted(set(candidate) - set(baseline))

    if only_baseline:
        print(f"NOTE: {len(only_baseline)} file(s) only in baseline, skipped: {only_baseline}")
    if only_candidate:
        print(f"NOTE: {len(only_candidate)} file(s) only in candidate, skipped: {only_candidate}")
    if not shared_files:
        raise SystemExit("No source_file names in common between the two inputs -- nothing to compare.")

    mismatches = []
    field_totals = defaultdict(int)
    field_matches = defaultdict(int)
    baseline_flagged = 0
    candidate_flagged = 0
    total_fields_compared = 0
    total_matches = 0

    for source_file in shared_files:
        b_rec, c_rec = baseline[source_file], candidate[source_file]
        baseline_flagged += len(b_rec.get("low_confidence_fields", []))
        candidate_flagged += len(c_rec.get("low_confidence_fields", []))

        for field_key in ALL_FIELD_KEYS:
            b_val = normalize(get_nested(b_rec, field_key))
            c_val = normalize(get_nested(c_rec, field_key))
            field_totals[field_key] += 1
            total_fields_compared += 1
            if b_val == c_val:
                field_matches[field_key] += 1
                total_matches += 1
            else:
                mismatches.append({
                    "source_file": source_file,
                    "field_key": field_key,
                    "section": "daily_function" if field_key in DAILY_FUNCTION_KEYS else field_key.split(".")[0],
                    "baseline_value": get_nested(b_rec, field_key),
                    "candidate_value": get_nested(c_rec, field_key),
                })

    # ---- Write mismatch CSV for spot-checking ----
    mismatch_path = Path(f"{args.out_prefix}_mismatches.csv")
    with mismatch_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "field_key", "section", "baseline_value", "candidate_value"])
        writer.writeheader()
        writer.writerows(mismatches)

    # ---- Per-field agreement summary ----
    field_summary = sorted(
        ((k, field_matches[k], field_totals[k]) for k in field_totals),
        key=lambda x: (x[1] / x[2], x[0]),
    )

    print(f"\nCompared {len(shared_files)} form(s), {len(ALL_FIELD_KEYS)} fields each "
          f"({total_fields_compared} total field comparisons).")
    print(f"Overall agreement: {total_matches}/{total_fields_compared} "
          f"({100 * total_matches / total_fields_compared:.1f}%)")

    daily_function_total = sum(field_totals[k] for k in DAILY_FUNCTION_KEYS)
    daily_function_matches = sum(field_matches[k] for k in DAILY_FUNCTION_KEYS)
    if daily_function_total:
        print(f"Daily Function agreement specifically: {daily_function_matches}/{daily_function_total} "
              f"({100 * daily_function_matches / daily_function_total:.1f}%) "
              "-- the hardest section for both approaches, worth watching closely.")

    print(f"\nFields flagged low-confidence/uncertain by each approach across all forms "
          f"(lower = less manual review left behind):")
    print(f"  Baseline (Cloud Vision): {baseline_flagged}")
    print(f"  Candidate (Gemini):      {candidate_flagged}")

    print(f"\nWorst-agreement fields (bottom 10) -- these are the ones most worth spot-checking "
          f"against the actual scans:")
    for field_key, matches, total in field_summary[:10]:
        print(f"  {field_key}: {matches}/{total} ({100 * matches / total:.0f}%)")

    print(f"\nWrote {len(mismatches)} mismatches to {mismatch_path} for row-by-row review.")


if __name__ == "__main__":
    main()
