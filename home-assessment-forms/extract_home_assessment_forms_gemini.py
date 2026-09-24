#!/usr/bin/env python3
"""
Single-pass home-assessment form extraction via Gemini's native PDF/document understanding
(Vertex AI). This is now the primary extraction path -- per the pilot bake-off and the 30-file
manual spot-check against ground truth (see OCR_Extraction_Spot_Check.xlsx / the project memo),
Gemini alone outperformed the Cloud Vision OCR baseline and the Document AI parser on every
section except Daily Function, so the two-way comparison in compare_extraction_outputs.py /
run_ocr_llm_pipeline.py is no longer how this runs day to day. Those scripts still exist for
reference if a future bake-off is needed.

Scope, per product decision: items 1-42 and 58 are extracted. Daily Function (43-57) is skipped
entirely -- the spot-check found it to be the one section where Gemini's accuracy dropped well
below every reasonable review threshold (most of its 15 rating columns matched ground truth only
65-90% of the time), so rather than extract it unreliably, this pipeline leaves it out and it
stays a manual-entry section until a better approach (cropped table region, gemini-2.5-pro,
self-consistency voting -- see the memo's Next Steps) is validated.

Output uses the same field keys and nested JSON shape as the Document AI script (header.*,
environmental_social.*, behavior.*, environmental_social.*, final_observations) -- no
daily_function.* keys anymore.

Setup:
  pip install google-genai
  Vertex AI project/location with Gemini access (already covered by your GCP BAA).

Usage:
    python extract_home_assessment_forms_gemini.py \\
        --input-dir ./home-assessment-forms \\
        --project-id your-gcp-project \\
        --location us-central1 \\
        --out-prefix ./home_assessment_extract_gemini \\
        [--model gemini-2.5-flash] [--workers 5] [--limit 10]

--input-dir is searched recursively (rglob), so pointing it at a parent folder containing
multiple scan subfolders (pilot_scans_1/, pilot_scans_2/, ...) picks up every PDF underneath in
one run instead of one folder at a time.
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:
    sys.exit("Missing dependency: pip install google-genai google-api-core")

# Reuse the exact same field definitions as the Document AI script so both outputs land on
# identical field keys. (DAILY_FUNCTION_LABELS is intentionally not imported -- this pipeline
# skips that section entirely; see the module docstring.)
from extract_home_assessment_forms_docai import (
    QUESTION_TEXT, BOOLEAN_FIELDS, CHOICE_FIELDS, set_nested, flatten_for_csv,
)

RATE_LIMIT_ERRORS = (ResourceExhausted, TooManyRequests)

PROMPT = """You are extracting structured data from a scanned home-assessment form (a home
health/safety questionnaire filled out during a caregiver visit). Read the entire document,
including any handwritten circles, checkmarks, or filled boxes indicating answers.

Fill in every field defined in the response schema. Each field's description tells you the
exact question it corresponds to on the form. For Yes/No questions, answer true or false only
based on what's actually marked -- if truly nothing is marked or the field is blank, use null
rather than guessing.

Ignore the "Daily Function" table (the section with ~15 activities rated across 5 columns) --
it is intentionally not part of the response schema and should not be extracted.

If a field is present on the form but you're genuinely not confident in your reading of it
(faint mark, ambiguous handwriting, ambiguous which of two options was intended), still give
your best answer, but also add its field key (the exact key from the schema, e.g.
"behavior.sleep_problems") to the uncertain_fields list so it can be flagged for human review
rather than silently trusted."""


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


def extract_with_retry(client, model, pdf_path, schema, max_attempts=5) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            return extract_one_form(client, model, pdf_path, schema)
        except RATE_LIMIT_ERRORS:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)  # 2, 4, 8, 16s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                     help="Searched recursively -- point this at a parent folder containing "
                          "multiple scan subfolders to process all of them in one run.")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--work-dir", default="work_gemini_extract",
                     help="Per-form checkpoint directory -- a form already extracted here is "
                          "skipped on re-run, so an interrupted batch resumes instead of "
                          "restarting and re-paying for every form.")
    ap.add_argument("--workers", type=int, default=5,
                     help="Forms processed concurrently. Default 5, matching the "
                          "run_ocr_llm_pipeline.py convention -- keep modest to avoid tripping "
                          "Vertex AI's per-project rate limits; calls retry automatically on 429s.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        sys.exit(f"No PDFs found under {input_dir}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
    schema = build_flat_schema()

    def checkpoint_path(pdf_path: Path) -> Path:
        # source_file dedupes on name alone (matches downstream key), so the checkpoint filename
        # does too -- two same-named PDFs in different subfolders would collide here on purpose,
        # since the rest of the pipeline (flatten_for_csv, compare scripts) key on filename too.
        return work_dir / f"{pdf_path.stem}.json"

    records = []
    to_process = []
    for p in pdf_paths:
        cp = checkpoint_path(p)
        if cp.exists():
            records.append(json.loads(cp.read_text()))
        else:
            to_process.append(p)

    print(f"Found {len(pdf_paths)} PDF(s) under {input_dir}: {len(records)} already extracted "
          f"(checkpoint hit), {len(to_process)} to process now ({args.workers} concurrent, "
          f"model={args.model}).")

    def process_one(pdf_path: Path):
        record = extract_with_retry(client, args.model, pdf_path, schema)
        checkpoint_path(pdf_path).write_text(json.dumps(record, indent=2))
        return pdf_path, record

    failed = []
    if to_process:
        done_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, p): p for p in to_process}
            for future in as_completed(futures):
                pdf_path = futures[future]
                done_count += 1
                try:
                    _, record = future.result()
                except Exception as err:  # noqa: BLE001
                    print(f"[{done_count}/{len(to_process)}] {pdf_path.name} ... FAILED ({err})")
                    failed.append((pdf_path.name, str(err)))
                    continue
                print(f"[{done_count}/{len(to_process)}] {pdf_path.name} ... ok "
                      f"({len(record['low_confidence_fields'])} flagged uncertain)")
                records.append(record)

    order = {p.name: i for i, p in enumerate(pdf_paths)}
    records.sort(key=lambda r: order.get(r["source_file"], 999999))

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

    if failed:
        print(f"\n{len(failed)} form(s) failed and were skipped: {[f[0] for f in failed]}")

    print(f"\nDone. {len(records)} form(s) processed.")
    print(f"Wrote {out_json_path} and {out_csv_path}.")


if __name__ == "__main__":
    main()
