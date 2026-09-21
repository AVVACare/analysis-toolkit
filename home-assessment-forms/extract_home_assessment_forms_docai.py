"""
Extracts scanned home assessment forms into CSV/JSON using Google Cloud
Document AI's Form Parser -- for use once your org's GCP Business Associate
Agreement is confirmed to cover Document AI, so PHI can flow through it.

How this differs from the other two scripts in this repo:
  - extract_home_assessment_forms.py        -> Anthropic Claude vision API
  - extract_home_assessment_forms_local.py  -> fully local, no API at all
  - extract_home_assessment_forms_docai.py  -> this one, via GCP Document AI

LAYOUT NOTE (found by testing against a real form): this form is one
continuous 4-column table running the whole document: item number |
question | Answer | Comments. Document AI reports this under
`document.pages[].tables`, separate from plain key-value form fields.

Two things learned from a real --debug run against an actual scan, both
now handled below:
  1. The printed item number is frequently garbled by OCR ("19\\n17",
     "5555\\n51", or an outright wrong digit like item 6's row reading
     "9"). The QUESTION TEXT in column 2, by contrast, reads cleanly. So
     this version matches each row against known question wording
     (fuzzy match) rather than trusting the item number -- far more
     reliable in practice, even though it looks less "exact" on paper.
  2. The answer lives in column 3 ("Answer"), not the last column
     ("Comments", which is usually blank or holds a follow-up note). An
     earlier version of this script read the wrong column.

KNOWN LIMITATION -- Daily Function (items 43-57) cannot be fully solved
this way. That section is printed as 5 separate rating columns
(Completely Independent / Requires Cueing or Coaching / Requires
Assistance / Cannot Do At All / Cannot Assess), but Document AI's table
detector collapsed all of them into the same generic 4-column template
used everywhere else on this form -- a checkmark that was drawn under, say,
the 3rd rating column shows up as content in either the generic "Answer"
or "Comments" cell depending on where it happened to land, with no
reliable way to recover which of the 5 categories it came from once that
happens. This script still records THAT something was marked (and what
raw text/symbol was found) but does not guess which category, and flags
every Daily Function row for manual review. If this section matters as
much as the others, it's worth reviewing those 15 rows by hand per form,
or handling just this section with a vision-based read (Claude), since
that's a spatial/visual judgment Document AI isn't preserving here.

One-time setup (do this before running against real PHI):
  1. Confirm with whoever owns your GCP BAA that Document AI is on the
     in-scope services list for your account.
  2. Enable the Document AI API on your GCP project.
  3. Create a "Form Parser" processor: GCP Console -> Document AI ->
     Create Processor -> Form Parser. Note the Processor ID and the
     location (e.g. "us" or "eu") it was created in.
  4. `pip install google-cloud-documentai`
  5. Authenticate: `gcloud auth application-default login`, or set
     GOOGLE_APPLICATION_CREDENTIALS to a service account key with the
     Document AI API User role.

Usage:
    python extract_home_assessment_forms_docai.py \\
        --input-dir ./scans \\
        --project-id your-gcp-project \\
        --location us \\
        --processor-id your-form-parser-processor-id \\
        --out-prefix ./home_assessment_extract_docai \\
        [--debug]
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

# NOTE: google-cloud-documentai is only imported lazily, inside process_one_form() and main()
# below -- not at module level. This module's field definitions (QUESTION_TEXT, BOOLEAN_FIELDS,
# normalize_boolean, set_nested, flatten_for_csv, etc.) are imported by every other script in
# this pipeline (the Cloud Vision and Gemini extractors, the comparison script, the
# orchestrator) even though Document AI itself is no longer part of the active pipeline -- a
# module-level import here would force everyone to install google-cloud-documentai just to get
# these shared constants.


# field_key -> question wording to fuzzy-match against column 2 of each
# table row. Kept close to the actual printed wording for best match
# quality; matching is fuzzy so minor OCR noise (missing punctuation,
# stray characters) is fine.
#
# Item numbers in the comments below refer to the form's own printed numbering
# (1-58 total: 1-42 are these header/home-safety/environmental/behavior questions,
# 43-57 are Daily Function -- see DAILY_FUNCTION_LABELS below -- and 58 is
# final_observations, the last item on the form).
_ALL_QUESTION_TEXT = {
    "header.in_home_assessment_date": "In-Home Assessment Date:",                    # 1
    "header.referral_date": "Referral Date:",                                        # 2
    "header.performed_by": "Performed by:",                                          # 3
    "header.patient_name": "Name of Patient:",                                       # 4
    "header.date_of_birth": "Date of Birth:",                                        # 5
    "header.medicare_id": "Medicare Identification Number:",                         # 6
    "header.address": "Address:",                                                    # 7
    "header.answers_provided_by": "Answers provided by:",                            # 8
    "header.living_situation": "Does the patient live in Single Family Home, Apartment, Assisted Living, or other?",  # 9
    "header.rent_or_own": "Does the patient rent or own?",                           # 10
    "header.living_arrangement_details": "Overall living arrangement who lives with the patient any pets stairs in the home where is patient's bedroom",  # 11
    "home_safety.bed_transfer_ability": "How well does the patient get in and out of bed",  # 12
    "home_safety.navigates_home_independently": "Is the patient able to navigate the home Independently",  # 13
    "home_safety.walkways_clear_of_hazards": "Are walkways clear of tripping hazards, such as throw rugs, cords, and excessive clutter",  # 14
    "home_safety.lighting": "Is the lighting in the home fairly bright or more on the dim side",  # 15
    "home_safety.stairs_clear_and_lit": "If stairs are accessible, are they free of clutter and have bright lighting",  # 16
    "home_safety.bathroom_has_mobility_supports": "Is the bathroom equipped with mobility supports, such as grab bars, shower chair, toilet riser",  # 17
    "home_safety.kitchen_access": "Does the patient have access to the kitchen",      # 18
    "home_safety.stove_has_safety_features": "Have safety knobs or an automatic shut-off switch been installed on the stove",  # 19
    "home_safety.medications_clearly_labeled": "Are medications clearly labeled",     # 20
    "home_safety.hazardous_items_securely_stored": "Are potentially hazardous items, such as medication, alcohol, cleaning products, matches, sharp objects, and power tools securely stored",  # 21
    "home_safety.firearms_securely_stored": "If there are firearms, are they safely and securely stored away from the patient, with ammunition stored separately",  # 22
    "home_safety.co_smoke_detectors_working": "Are carbon monoxide detectors and smoke detectors installed and working properly",  # 23
    "home_safety.fire_extinguisher_present": "Is there a fire extinguisher in the home",  # 24
    "home_safety.patient_wandered_before": "Has the patient ever wandered out of the home",  # 25
    "home_safety.wandering_mitigations_in_place": "If previous wandering or concerns about wandering, have safety measures such as locks, cameras, or other strategies been set up",  # 26
    "home_safety.patient_still_driving": "Is the patient still driving",              # 27
    "home_safety.things_missing_that_could_help": "Do you feel as though there are things missing in the home that could be helpful",  # 28
    "environmental_social.socially_isolated": "Is the patient socially isolated outside of their normal behaviors",  # 29
    "environmental_social.targeted_for_exploitation": "Has the patient been targeted for exploitation or scamming",  # 30
    "environmental_social.housing_insecurity_risk": "Is the patient at risk for housing insecurity",  # 31
    "environmental_social.food_insecurity_risk": "Is the patient at risk for food insecurity or poor nutrition",  # 32
    "environmental_social.fallen_in_last_year": "Has the patient fallen in the last year",  # 33
    "environmental_social.other_observations": "Other Environmental/Social Observations",  # 34
    "behavior.agitation_aggression": "Agitation/Aggression: Does the patient get angry or hostile",  # 35
    "behavior.hallucinations": "Hallucinations: Does the patient see and/or hear things that no one else can see or hear",  # 36
    "behavior.irritability_moodiness": "Irritability/Moodiness: Does the patient act impatient",  # 37
    "behavior.suspiciousness_paranoia": "Suspiciousness/Paranoia: Is the patient suspicious without good reason",  # 38
    "behavior.indifference_withdrawal": "Indifference/Social Withdrawal: Does the patient seem less interested in their usual activities",  # 39
    "behavior.sleep_problems": "Sleep Problems: Does the patient have trouble sleeping at night",  # 40
    "behavior.anxiety_shadowing": "Anxiety/Shadowing: Does the patient routinely follow around their caregiver",  # 41
    "behavior.other_observations": "Other Behavioral Observations",                  # 42
    "final_observations": "Any other observations or concerns",                      # 58 (last item on the form)
}

# Per product decision: only items 1-8 and 29-through-end are wanted. Items 9-28 --
# the "living situation" header fields (9-11) and the entire Home Safety section
# (12-28, grab bars/smoke detectors/firearms storage/etc.) -- are excluded. Kept as
# _ALL_QUESTION_TEXT above (rather than deleted outright) so re-enabling any of these
# later is a one-line change instead of re-typing the question wording from scratch.
_EXCLUDED_ITEM_KEYS = {
    "header.living_situation", "header.rent_or_own", "header.living_arrangement_details",
} | {k for k in _ALL_QUESTION_TEXT if k.startswith("home_safety.")}

QUESTION_TEXT = {k: v for k, v in _ALL_QUESTION_TEXT.items() if k not in _EXCLUDED_ITEM_KEYS}

# Rows that are section headers/titles rather than questions -- if a row's
# best fuzzy match score doesn't clear this bar against anything in
# QUESTION_TEXT, it's treated as noise (a section title, a repeated
# "Field # / Field / Answer / Comments" header row, etc.) rather than
# forced into the nearest (wrong) question.
MATCH_THRESHOLD = 0.45

# Rows that are pure boilerplate -- repeated table headers or section
# title bands -- get skipped before any fuzzy matching is attempted, so
# they can never falsely claim a real field's match slot.
JUNK_ROW_MARKERS = {"field #", "field", "answer", "comments", "home safety",
                     "environmental", "behavior", "daily function"}


def is_junk_row(cells: list) -> bool:
    normalized = [c.strip().lower().rstrip(":?") for c in cells[:2]]
    return any(c in JUNK_ROW_MARKERS for c in normalized)

DAILY_FUNCTION_LABELS = {
    "daily_function.use_telephone": "Use the Telephone",
    "daily_function.shopping": "Shopping",
    "daily_function.laundry": "Laundry",
    "daily_function.driving_or_public_transportation": "Driving or using public transportation",
    "daily_function.food_preparation": "Food Preparation",
    "daily_function.medication_management": "Medication Management",
    "daily_function.housekeeping": "Housekeeping",
    "daily_function.finances": "Finances",
    "daily_function.bathing": "Bathing",
    "daily_function.dressing": "Dressing (including getting clothes from drawers or closet)",
    "daily_function.toileting": "Toileting (including cleaning, redressing, washing hands)",
    "daily_function.transferring": "Transferring in and out of a chair or bed",
    "daily_function.continence": "Continence (control of bladder and bowels)",
    "daily_function.feeding": "Feeding",
    "daily_function.grooming": "Grooming (including teeth, hair, shaving, etc.)",
}

_ALL_BOOLEAN_FIELDS = {
    "home_safety.navigates_home_independently", "home_safety.walkways_clear_of_hazards",
    "home_safety.stairs_clear_and_lit", "home_safety.bathroom_has_mobility_supports",
    "home_safety.kitchen_access", "home_safety.stove_has_safety_features",
    "home_safety.medications_clearly_labeled", "home_safety.hazardous_items_securely_stored",
    "home_safety.firearms_securely_stored", "home_safety.co_smoke_detectors_working",
    "home_safety.fire_extinguisher_present", "home_safety.patient_wandered_before",
    "home_safety.wandering_mitigations_in_place", "home_safety.patient_still_driving",
    "environmental_social.socially_isolated", "environmental_social.targeted_for_exploitation",
    "environmental_social.housing_insecurity_risk", "environmental_social.food_insecurity_risk",
    "environmental_social.fallen_in_last_year", "behavior.agitation_aggression",
    "behavior.hallucinations", "behavior.irritability_moodiness", "behavior.suspiciousness_paranoia",
    "behavior.indifference_withdrawal", "behavior.sleep_problems", "behavior.anxiety_shadowing",
}
BOOLEAN_FIELDS = {k for k in _ALL_BOOLEAN_FIELDS if k not in _EXCLUDED_ITEM_KEYS}

_ALL_CHOICE_FIELDS = {
    "header.living_situation": ["single_family_home", "apartment", "assisted_living", "other"],
    "header.rent_or_own": ["rent", "own"],
    "home_safety.bed_transfer_ability": ["independent", "requires_assistance", "full_assistance"],
    "home_safety.lighting": ["bright", "dim"],
}
CHOICE_FIELDS = {k: v for k, v in _ALL_CHOICE_FIELDS.items() if k not in _EXCLUDED_ITEM_KEYS}

YES_TOKENS = {"y", "yes"}
NO_TOKENS = {"n", "no"}


def get_text(text_anchor, full_text: str) -> str:
    """Document AI reports text as offsets into the page's full text blob rather than inline strings."""
    if not text_anchor or not text_anchor.text_segments:
        return ""
    parts = []
    for seg in text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index else 0
        end = int(seg.end_index)
        parts.append(full_text[start:end])
    return "".join(parts).strip()


def normalize_boolean(text: str) -> tuple:
    token = text.strip().lower()
    if not token:
        return None, "low"
    first_word = re.split(r"[\s.:,]+", token)[0]
    if first_word in YES_TOKENS:
        return True, "ok"
    if first_word in NO_TOKENS:
        return False, "ok"
    if "n/a" in token:
        return None, "ok"  # explicitly not applicable -- a real answer, not a missing one
    return None, "low"


def normalize_choice(text: str, options: list) -> tuple:
    token = text.strip().lower()
    for opt in options:
        if opt.replace("_", " ") in token or opt in token:
            return opt, "ok"
    return None, "low"


def set_nested(d: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def partial_ratio(phrase: str, text: str) -> float:
    """How much of `phrase` appears (in order) inside `text`, normalized by phrase length only.

    Plain difflib.ratio() penalizes a correct match whenever the OCR'd cell
    has extra trailing content (e.g. a sub-bullet like "-If yes, are there
    any concerns...") because it's sensitive to *combined* string length --
    a long row can end up scoring lower against its own correct question
    than against a totally unrelated short one. This instead asks "how much
    of the canonical question text is actually contained in this cell,"
    which is what we actually want to know.
    """
    phrase, text = phrase.lower(), text.lower()
    if not phrase:
        return 0.0
    sm = difflib.SequenceMatcher(None, phrase, text)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(phrase)


def best_match(text: str, label_dict: dict, threshold: float = MATCH_THRESHOLD):
    best_key, best_score = None, 0.0
    for key, question in label_dict.items():
        score = partial_ratio(question, text)
        if score > best_score:
            best_key, best_score = key, score
    if best_score >= threshold:
        return best_key, best_score
    return None, best_score


def match_daily_function_label(text: str):
    """Daily Function labels are single short words ("Shopping", "Bathing").

    partial_ratio (normalized by the *short* label's length) massively
    over-scores short labels against long unrelated text -- a handful of
    coincidentally shared letters is enough to look like a near-perfect
    match. These are printed as clean standalone words in their own cell,
    so an exact (whitespace/punctuation-insensitive) match is both the
    common case and by far the safest one; a conservative plain-ratio
    fallback only catches minor OCR noise, not loose partial containment.
    """
    normalized_text = re.sub(r"[^a-z]", "", text.lower())
    best_key, best_score = None, 0.0
    for key, label in DAILY_FUNCTION_LABELS.items():
        normalized_label = re.sub(r"[^a-z]", "", label.lower())
        if normalized_text == normalized_label:
            return key, 1.0
        score = difflib.SequenceMatcher(None, normalized_text, normalized_label).ratio()
        if score > best_score:
            best_key, best_score = key, score
    if best_score >= 0.85:
        return best_key, best_score
    return None, best_score


def row_bbox(row) -> tuple:
    """Union bounding box (normalized 0-1 coords) across every cell in a table row.

    Used to crop the Daily Function section out of the page image for the Gemini
    correction pass (see gemini_correction_pass.py) -- Document AI's table detector
    collapses the 5 rating columns into whatever generic columns it expects, but the
    *pixels* still show the real 5-column layout, so cropping the row region and handing
    it to a vision model sidesteps the column-collapse problem entirely instead of trying
    to recover it from Document AI's (already-lost) column structure.
    """
    xs, ys = [], []
    for cell in row.cells:
        poly = cell.layout.bounding_poly
        if not poly or not poly.normalized_vertices:
            continue
        for v in poly.normalized_vertices:
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def parse_tables(document, full_text: str, record: dict, low_confidence_fields: list,
                  daily_function_review: dict, raw_rows_out: list = None,
                  daily_function_crop_regions: dict = None):
    matched_daily_function_keys = set()
    matched_field_keys = set()  # first genuine match to a field wins; a later
    # coincidental match (e.g. a different row's leftover text scoring just
    # above threshold) can no longer silently overwrite a correct value.

    for page_number, page in enumerate(document.pages):
        for table in page.tables:
            all_rows = list(table.header_rows) + list(table.body_rows)
            for row in all_rows:
                cells = [get_text(cell.layout.text_anchor, full_text) for cell in row.cells]
                if raw_rows_out is not None:
                    raw_rows_out.append(cells)
                if len(cells) < 2 or is_junk_row(cells):
                    continue

                question_text = cells[1]

                df_key, df_score = match_daily_function_label(question_text)
                if df_key is not None and df_key not in matched_daily_function_keys:
                    matched_daily_function_keys.add(df_key)
                    # Which extra cell(s) (beyond item#/question) have any content at all --
                    # tells us *that* something was marked, not reliably *which* of the 5
                    # rating categories, since Document AI collapsed those columns. Recording
                    # the raw text so a human reviewer doesn't have to re-open the scan.
                    marked_cells = [c for c in cells[2:] if c.strip()]
                    set_nested(record, df_key, None)
                    low_confidence_fields.append(df_key)
                    daily_function_review[df_key] = marked_cells if marked_cells else None

                    if daily_function_crop_regions is not None:
                        bbox = row_bbox(row)
                        if bbox is not None:
                            x0, y0, x1, y1 = bbox
                            existing = daily_function_crop_regions.get(page_number)
                            if existing is None:
                                daily_function_crop_regions[page_number] = [x0, y0, x1, y1]
                            else:
                                daily_function_crop_regions[page_number] = [
                                    min(existing[0], x0), min(existing[1], y0),
                                    max(existing[2], x1), max(existing[3], y1),
                                ]
                    continue

                field_key, score = best_match(question_text, QUESTION_TEXT)
                if field_key is None or field_key in matched_field_keys:
                    continue  # section title, repeated table header row, unrecognized text,
                    # or a field we already have a genuine match for
                matched_field_keys.add(field_key)

                answer_text = cells[2].strip() if len(cells) > 2 else ""
                comments_text = cells[3].strip() if len(cells) > 3 else ""
                raw_combined = " ".join(t for t in [answer_text, comments_text] if t)

                if field_key in BOOLEAN_FIELDS:
                    value, confidence = normalize_boolean(answer_text)
                elif field_key in CHOICE_FIELDS:
                    value, confidence = normalize_choice(answer_text, CHOICE_FIELDS[field_key])
                else:
                    value, confidence = (raw_combined if raw_combined else None), "ok"

                set_nested(record, field_key, value)
                if confidence == "low" or score < 0.6:
                    low_confidence_fields.append(field_key)
                    if value is None and raw_combined:
                        # Don't just say "we couldn't parse this" -- keep the
                        # actual text so a reviewer isn't sent back to the
                        # original scan for something we already captured.
                        set_nested(record, field_key + "_raw_text", raw_combined)


def process_one_form(client, processor_name: str, pdf_path: Path, debug: bool = False) -> tuple:
    try:
        from google.cloud import documentai
    except ImportError:
        sys.exit("Missing dependency: pip install google-cloud-documentai")

    content = pdf_path.read_bytes()
    raw_document = documentai.RawDocument(content=content, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document
    full_text = document.text

    record: dict = {"source_file": pdf_path.name}
    low_confidence_fields: list = []
    daily_function_review: dict = {}
    raw_rows: list = [] if debug else None
    daily_function_crop_regions: dict = {}

    parse_tables(document, full_text, record, low_confidence_fields, daily_function_review, raw_rows,
                 daily_function_crop_regions)

    record["daily_function_needs_manual_review"] = daily_function_review
    record["low_confidence_fields"] = low_confidence_fields
    # page_number -> [x0,y0,x1,y1] normalized (0-1) crop region covering the Daily Function
    # table on that page. Consumed by gemini_correction_pass.py to crop the actual page
    # image (not just the OCR'd text) and ask a vision model to read the 5-column ratings
    # directly off the pixels, since Document AI's table parser doesn't preserve them.
    record["daily_function_crop_regions"] = {
        str(k): v for k, v in daily_function_crop_regions.items()
    }

    debug_info = None
    if debug:
        debug_info = {"source_file": pdf_path.name, "raw_table_rows": raw_rows}

    return record, debug_info


def flatten_for_csv(record: dict) -> dict:
    flat = {}

    def _walk(prefix: str, value):
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, list):
            flat[prefix] = json.dumps(value) if value and isinstance(value[0], dict) else "; ".join(str(v) for v in value)
        else:
            flat[prefix] = value

    _walk("", record)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder containing one PDF per form")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--location", required=True, help="Document AI processor location, e.g. 'us' or 'eu'")
    parser.add_argument("--processor-id", required=True, help="Form Parser processor ID")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix (writes {prefix}.json and {prefix}.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files (pilot batch)")
    parser.add_argument("--debug", action="store_true", help="Also write {out-prefix}_debug_tables.json with every raw table row exactly as OCR'd")
    args = parser.parse_args()

    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai
    except ImportError:
        sys.exit("Missing dependency: pip install google-cloud-documentai")

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        sys.exit(f"No PDFs found in {input_dir}")

    opts = ClientOptions(api_endpoint=f"{args.location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(args.project_id, args.location, args.processor_id)

    print(f"Found {len(pdf_paths)} PDF(s). Extracting via Document AI (question-text matching)...")

    records = []
    debug_records = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"  [{i}/{len(pdf_paths)}] {pdf_path.name} ...", end=" ", flush=True)
        try:
            record, debug_info = process_one_form(client, processor_name, pdf_path, debug=args.debug)
        except Exception as err:  # noqa: BLE001
            print(f"FAILED ({err})")
            continue
        print(f"ok ({len(record['low_confidence_fields'])} low-confidence)")
        records.append(record)
        if debug_info is not None:
            debug_records.append(debug_info)

    out_json_path = Path(f"{args.out_prefix}.json")
    out_csv_path = Path(f"{args.out_prefix}.csv")
    out_json_path.write_text(json.dumps(records, indent=2))

    if args.debug:
        debug_path = Path(f"{args.out_prefix}_debug_tables.json")
        debug_path.write_text(json.dumps(debug_records, indent=2))
        print(f"Wrote raw table rows to {debug_path}.")

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
    print(
        "\nCheck 'low_confidence_fields' for Y/N or choice answers that didn't match a known token. "
        "Check 'daily_function_needs_manual_review' for every Daily Function item -- Document AI's table "
        "detection doesn't reliably preserve which of the 5 rating columns was marked for this form, so "
        "those 15 items need a human look (the raw cell text captured there is a hint, not the answer)."
    )


if __name__ == "__main__":
    main()
