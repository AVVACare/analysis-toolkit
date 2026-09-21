#!/usr/bin/env python3
"""
Single-pass home-assessment form extraction via Gemini's native PDF/document understanding
(Vertex AI) -- the "just send the whole form to a vision model" alternative to the
Document AI + targeted-correction hybrid in extract_home_assessment_forms_docai.py /
gemini_correction_pass.py.

This is meant to run side-by-side against the Cloud Vision baseline on the SAME set of forms,
so compare_extraction_outputs.py can show where they agree and where they don't -- a real
bake-off, not a replacement, until the comparison says otherwise.

Why this can work in one pass where Document AI needed a second correction step: Gemini
takes the PDF directly (native vision over the actual page, not a table-structure parse) and
reads the whole form -- including the Daily Function section's 5 rating columns -- in the
same call, since "which column has the mark" is a visual judgment a multimodal model handles
natively rather than something that has to survive an intermediate table-detection step.

Output uses the exact same field keys and nested JSON shape as the Document AI script
(header.*, environmental_social.*, behavior.*, daily_function.*, final_observations)
specifically so the two outputs are directly comparable field-by-field. Only items 1-8 and
29-through-end are extracted, per product decision -- items 9-28 (living-situation header
fields and the entire Home Safety section) are intentionally skipped, same as the Document AI
script (see _EXCLUDED_ITEM_KEYS there).

Setup:
  pip install google-genai
  Vertex AI project/location with Gemini access (already covered by your GCP BAA).

Usage:
    python extract_home_assessment_forms_gemini.py \\
        --input-dir ./scans \\
        --project-id your-gcp-project \\
        --location us-central1 \\
        --out-prefix ./home_assessment_extract_gemini \\
        [--model gemini-2.5-pro] [--limit 10]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Missing dependency: pip install google-genai")

# Reuse the exact same field definitions as the Document AI script so both outputs land on
# identical field keys -- that's what makes compare_extraction_outputs.py a fair comparison
# rather than an apples-to-oranges one.
from extract_home_assessment_forms_docai import (
    QUESTION_TEXT, BOOLEAN_FIELDS, CHOICE_FIELDS, DAILY_FUNCTION_LABELS,
    set_nested, flatten_for_csv,
)

RATING_OPTIONS = [
    "completely_independent",
    "requires_cueing_or_coaching",
    "requires_assistance",
    "cannot_do_at_all",
    "cannot_assess",
]

PROMPT = """You are extracting structured data from a scanned home-assessment form (a home
health/safety questionnaire filled out during a caregiver visit). Read the entire document,
including any handwritten circles, checkmarks, or filled boxes indicating answers.

Fill in every field defined in the response schema. Each field's description tells you the
exact question it corresponds to on the form. For Yes/No questions, answer true or false only
based on what's actually marked -- if truly nothing is marked or the field is blank, use null
rather than guessing.

Pay special attention to the "Daily Function" section: it's a table of ~15 activities (Use
the Telephone, Shopping, Bathing, etc.), each with 5 rating columns (Completely Independent,
Requires Cueing or Coaching, Requires Assistance, Cannot Do At All, Cannot Assess). Read
which single column is marked for each row directly off the page -- this is a visual judgment
about which column has the mark, not a text-matching task.

If a field is present on the form but you're genuinely not confident in your reading of it
(faint mark, ambiguous handwriting, ambiguous which of two options was intended), still give
your best answer, but also add its field key (the exact key from the schema, e.g.
"behavior.sleep_problems" or "daily_function.bathing") to the uncertain_fields list so it can
be flagged for human review rather than silently trusted."""


def build_flat_schema() -> dict:
    """One flat JSON Schema object, keyed by the exact dotted field-key strings used
    throughout the pipeline (e.g. "header.patient_name", "daily_function.bathing"). JSON
    Schema property names don't need to be valid identifiers, so we can use the dotted keys
    directly -- this lets us convert Gemini's response straight into the same nested record
    shape via set_nested(), with no key-remapping step to get wrong."""
    # NOTE: Vertex AI's Schema object is a stricter subset of JSON Schema than the Gemini API
    # docs' generic examples suggest -- "type" must be a single value from a fixed enum
    # (STRING/NUMBER/INTEGER/BOOLEAN/ARRAY/OBJECT/NULL), not a JSON-Schema-style union list
    # like ["string", "null"]. To allow null, use "nullable": True alongside a single "type"
    # instead. Enum lists must contain only real enum values (no literal None entries either).
    properties = {}
    for field_key, question in QUESTION_TEXT.items():
        if field_key in BOOLEAN_FIELDS:
            properties[field_key] = {"type": "BOOLEAN", "nullable": True, "description": question}
        elif field_key in CHOICE_FIELDS:
            options = CHOICE_FIELDS[field_key]
            properties[field_key] = {
                "type": "STRING", "enum": options, "nullable": True,
                "description": f"{question} (one of: {', '.join(options)})",
            }
        else:
            properties[field_key] = {"type": "STRING", "nullable": True, "description": question}

    for field_key, label in DAILY_FUNCTION_LABELS.items():
        properties[field_key] = {
            "type": "STRING", "enum": RATING_OPTIONS, "nullable": True,
            "description": f"Daily Function rating for: {label}",
        }

    properties["uncertain_fields"] = {
        "type": "ARRAY",
        "items": {"type": "STRING"},
        "description": "Field keys (from this same schema) where you weren't confident in "
                        "your reading and a human should double-check the original scan.",
    }

    return {"type": "OBJECT", "properties": properties}


def extract_one_form(client, model: str, pdf_path: Path, schema: dict) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    flat = json.loads(response.text)

    record: dict = {"source_file": pdf_path.name}
    uncertain_fields = flat.pop("uncertain_fields", []) or []
    for field_key, value in flat.items():
        set_nested(record, field_key, value)
    record["low_confidence_fields"] = uncertain_fields  # same key name as the DocAI script,
    # so compare_extraction_outputs.py can treat both the same way
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", default="gemini-2.5-flash",
                     help="gemini-2.5-flash is the default: pilot testing showed 100% agreement "
                          "with the Cloud Vision baseline at a fraction of gemini-2.5-pro's "
                          "latency. Pass gemini-2.5-pro if you want to re-check accuracy at scale.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files (pilot batch)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        sys.exit(f"No PDFs found in {input_dir}")

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
    schema = build_flat_schema()

    print(f"Found {len(pdf_paths)} PDF(s). Extracting via Gemini ({args.model}, native document understanding)...")

    records = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"  [{i}/{len(pdf_paths)}] {pdf_path.name} ...", end=" ", flush=True)
        try:
            record = extract_one_form(client, args.model, pdf_path, schema)
        except Exception as err:  # noqa: BLE001
            print(f"FAILED ({err})")
            continue
        print(f"ok ({len(record['low_confidence_fields'])} flagged uncertain)")
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
    print("\nNext: run compare_extraction_outputs.py against your Cloud Vision baseline output "
          "on the same files to see where the two approaches agree and disagree.")


if __name__ == "__main__":
    main()
