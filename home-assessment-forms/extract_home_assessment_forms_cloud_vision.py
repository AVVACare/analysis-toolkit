#!/usr/bin/env python3
"""
Home-assessment form extraction via Cloud Vision OCR -- adapter layer over
extract_assessment.py (the field-tested script from the earlier pilot run that outperformed
Document AI). This file does NOT reimplement any OCR/row-detection logic itself; it only:

  1. Calls extract_assessment.py's extract_rows_for_pdf() to get raw field_num-keyed rows
     ({field_num, field, response, needs_review}), using home_assessment_schema.json for the
     calibrated column layout.
  2. Maps each field_num to the dotted field_key naming used throughout the rest of this
     pipeline (header.*, environmental_social.*, behavior.*, daily_function.*,
     final_observations) -- the same keys extract_home_assessment_forms_docai.py and
     extract_home_assessment_forms_gemini.py use, so compare_extraction_outputs.py and
     run_ocr_llm_pipeline.py work with this as a drop-in baseline leg.
  3. Normalizes response text into the same value types the rest of the pipeline expects:
     True/False for Y/N fields (via the exact same normalize_boolean() used elsewhere), a
     snake_case rating key for Daily Function fields, and passthrough text otherwise.

home_assessment_schema.json currently covers items 1-8 and 29-through-58 (58 = final
observations), matching the earlier product decision to skip items 9-28 (the living-situation
header fields and the entire Home Safety section) -- consistent with the other two extraction
approaches. NOTE: items 1-8's column layout is inferred (assumed to share the same
"standard_table" columns as items 29-42, since the source form is one continuous table per
extract_home_assessment_forms_docai.py's own notes) rather than independently
pixel-calibrated the way items 29-58 were. Worth confirming on the first real run --check
whether items 1-8 come back with real values or all land in needs_review/low_confidence,
which would mean that assumption needs correcting against an actual scan.

Setup:
  pip install google-cloud-vision pdf2image pandas pillow numpy
  Also install poppler (pdf2image's PDF-rendering backend): macOS `brew install poppler`,
  Ubuntu/Debian `apt-get install poppler-utils`.
  Cloud Vision API enabled + your service account has access (same as extract_assessment.py's
  standalone requirements).

Usage:
    python extract_home_assessment_forms_cloud_vision.py \\
        --input-dir ./scans \\
        --schema ./home_assessment_schema.json \\
        --out-prefix ./home_assessment_extract_cloud_vision \\
        [--limit 10]
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

try:
    from google.cloud import vision
except ImportError:
    sys.exit("Missing dependency: pip install google-cloud-vision")

from extract_assessment import extract_rows_for_pdf
from extract_home_assessment_forms_docai import (
    BOOLEAN_FIELDS, normalize_boolean, set_nested, flatten_for_csv,
)

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "home_assessment_schema.json"

# field_num -> dotted field_key, matching extract_home_assessment_forms_docai.py's
# _ALL_QUESTION_TEXT / DAILY_FUNCTION_LABELS ordering exactly. Items 9-28 are intentionally
# absent (see module docstring and _EXCLUDED_ITEM_KEYS in extract_home_assessment_forms_docai.py).
FIELD_NUM_TO_KEY = {
    1: "header.in_home_assessment_date",
    2: "header.referral_date",
    3: "header.performed_by",
    4: "header.patient_name",
    5: "header.date_of_birth",
    6: "header.medicare_id",
    7: "header.address",
    8: "header.answers_provided_by",
    29: "environmental_social.socially_isolated",
    30: "environmental_social.targeted_for_exploitation",
    31: "environmental_social.housing_insecurity_risk",
    32: "environmental_social.food_insecurity_risk",
    33: "environmental_social.fallen_in_last_year",
    34: "environmental_social.other_observations",
    35: "behavior.agitation_aggression",
    36: "behavior.hallucinations",
    37: "behavior.irritability_moodiness",
    38: "behavior.suspiciousness_paranoia",
    39: "behavior.indifference_withdrawal",
    40: "behavior.sleep_problems",
    41: "behavior.anxiety_shadowing",
    42: "behavior.other_observations",
    43: "daily_function.use_telephone",
    44: "daily_function.shopping",
    45: "daily_function.laundry",
    46: "daily_function.driving_or_public_transportation",
    47: "daily_function.food_preparation",
    48: "daily_function.medication_management",
    49: "daily_function.housekeeping",
    50: "daily_function.finances",
    51: "daily_function.bathing",
    52: "daily_function.dressing",
    53: "daily_function.toileting",
    54: "daily_function.transferring",
    55: "daily_function.continence",
    56: "daily_function.feeding",
    57: "daily_function.grooming",
    58: "final_observations",
}

RATING_LABEL_TO_KEY = {
    "Completely Independent (1)": "completely_independent",
    "Requires Cueing or Coaching (2)": "requires_cueing_or_coaching",
    "Requires Assistance (3)": "requires_assistance",
    "Cannot Do At All (4)": "cannot_do_at_all",
    "Cannot Assess (5)": "cannot_assess",
}


def normalize_rating_response(text: str):
    if not text or text.startswith("NEEDS REVIEW"):
        return None
    # A row can legitimately have more than one rating marked (see
    # extract_assessment.py's detect_rating_from_ocr_digit) -- keep all of them rather than
    # picking one arbitrarily.
    labels = [t.strip() for t in text.split(";")]
    mapped = [RATING_LABEL_TO_KEY.get(label, label) for label in labels]
    return "; ".join(mapped)


def rows_to_record(source_file: str, rows: list) -> dict:
    record: dict = {"source_file": source_file}
    low_confidence_fields = []

    for row in rows:
        field_key = FIELD_NUM_TO_KEY.get(row["field_num"])
        if field_key is None:
            continue  # shouldn't happen if schema.json only contains mapped field_nums

        response = row["response"]
        if field_key in BOOLEAN_FIELDS:
            value, confidence = normalize_boolean(response or "")
            if confidence == "low":
                low_confidence_fields.append(field_key)
        elif field_key.startswith("daily_function."):
            value = normalize_rating_response(response)
        else:
            value = response if response and not str(response).startswith("NEEDS REVIEW") else None

        if row["needs_review"] and field_key not in low_confidence_fields:
            low_confidence_fields.append(field_key)

        set_nested(record, field_key, value)

    record["low_confidence_fields"] = low_confidence_fields
    return record


def process_one_form(vision_client, pdf_path: Path, dpi: int = None, schema: dict = None) -> tuple:
    """Same (record, debug_info) signature as the other extractors' process_one_form /
    extract_one_form, so run_ocr_llm_pipeline.py can use this as a drop-in baseline leg.
    `dpi` is accepted for interface compatibility but unused here -- extract_assessment.py
    renders at its own internal default (200 DPI) via pdf2image."""
    if schema is None:
        schema = json.loads(DEFAULT_SCHEMA_PATH.read_text())

    with tempfile.TemporaryDirectory() as img_dir:
        rows = extract_rows_for_pdf(str(pdf_path), schema, vision_client, img_dir)

    record = rows_to_record(pdf_path.name, rows)
    return record, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files (pilot batch)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        sys.exit(f"No PDFs found in {input_dir}")

    schema = json.loads(Path(args.schema).read_text())
    vision_client = vision.ImageAnnotatorClient()

    print(f"Found {len(pdf_paths)} PDF(s). Extracting via Cloud Vision OCR ...")

    records = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"  [{i}/{len(pdf_paths)}] {pdf_path.name} ...", end=" ", flush=True)
        try:
            record, _ = process_one_form(vision_client, pdf_path, schema=schema)
        except Exception as err:  # noqa: BLE001
            print(f"FAILED ({err})")
            continue
        print(f"ok ({len(record['low_confidence_fields'])} flagged for review)")
        records.append(record)

    out_json_path = Path(f"{args.out_prefix}.json")
    out_csv_path = Path(f"{args.out_prefix}.csv")
    out_json_path.write_text(json.dumps(records, indent=2))

    flat_records = [flatten_for_csv(r) for r in records]
    all_keys: list = []
    for r in flat_records:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flat_records)

    print(f"\nDone. {len(records)} form(s) processed.")
    print(f"Wrote {out_json_path} and {out_csv_path}.")
    print("\nCheck items 1-8 specifically on this first run -- their column layout was inferred "
          "from the rest of the table, not independently pixel-calibrated. If they're all "
          "coming back empty/needs_review, that assumption needs fixing against a real scan.")


if __name__ == "__main__":
    main()
