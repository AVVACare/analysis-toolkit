#!/usr/bin/env python3
"""
End-to-end home-assessment extraction pipeline: Cloud Vision OCR (row/column reconstruction +
question-text matching) first, then Gemini (native vision) second, then a cross-method
comparison that flags fields for human review based on DISAGREEMENT BETWEEN THE TWO SYSTEMS --
not based on either system's own self-reported confidence.

Why this exists: self-reported LLM/OCR confidence is not well-calibrated, especially on
handwriting -- a system can misread a messy mark and still not flag it as low-confidence,
because from its own point of view the reading looked fine. The fix isn't better prompting,
it's an independent check: run two systems with different failure modes (Cloud Vision's
geometry-based row reconstruction vs. Gemini's holistic vision read) and treat any field where
they DISAGREE as needing a human look, regardless of what either one says about its own
certainty. A field where both self-reported "confident" but the two answers don't match is
exactly the "confidently wrong, unflagged" case this is built to catch.

Each system's own low-confidence/uncertain flags are still recorded in the review worklist as
extra context (worth knowing if a field was *also* self-flagged), but they are NOT what
determines whether a field ends up in the worklist -- disagreement is.

Where the two systems agree, that value is trusted and written straight into the final record.
Where they disagree, the final record gets null for that field (not a guess at which system is
"right") and the field goes into the review worklist with both raw values side by side.

Usage:
    python run_ocr_llm_pipeline.py \\
        --input-dir ./scans \\
        --project-id avvacare-clinical-ops \\
        --gemini-location us-central1 \\
        --out-prefix ./pipeline_run \\
        [--vision-schema ./home_assessment_schema.json] [--gemini-model gemini-2.5-flash]
        [--workers 5] [--limit 10]

Outputs two files, both under --out-prefix, with the same four groupings:
    {prefix}.xlsx -- formatted workbook for human review, with four tabs:
        Baseline        -- raw Cloud Vision output, one row per form
        Gemini          -- raw Gemini output, one row per form
        Final           -- merged record per form: agreed values filled in, disagreements blank
        Review Worklist -- one row per disagreeing field, for human review, across all forms
    {prefix}.json -- the same four groupings as nested records (real null/bool types, dotted
        keys unflattened) for anyone scripting against the output instead of reviewing it by eye:
        {"baseline": [...], "gemini": [...], "final": [...], "review_worklist": [...]}

Runs up to --workers forms concurrently (default 5) -- each worker does that form's Cloud
Vision call then its Gemini call, so at --workers 5, five forms are in flight at once (ten
concurrent API calls: 5 Cloud Vision, 5 Gemini, though not perfectly synchronized since forms
finish their OCR and hand off to Gemini at different times). Kept modest by default because
Vertex AI's Gemini endpoints have per-project rate limits that vary by model/region -- pushing
too many concurrent forms risks 429 rate-limit errors. Gemini calls retry automatically on 429s
with exponential backoff (up to 5 attempts) before giving up on that form.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

try:
    from google.cloud import vision
except ImportError:
    sys.exit("Missing dependency: pip install google-cloud-vision")

try:
    from google import genai
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:
    sys.exit("Missing dependency: pip install google-genai")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency: pip install pandas openpyxl")

from extract_home_assessment_forms_docai import set_nested, flatten_for_csv, QUESTION_TEXT, DAILY_FUNCTION_LABELS
from extract_home_assessment_forms_cloud_vision import process_one_form as process_one_form_vision, FIELD_NUM_TO_KEY
from extract_home_assessment_forms_gemini import extract_one_form, build_flat_schema
from compare_extraction_outputs import ALL_FIELD_KEYS, DAILY_FUNCTION_KEYS, normalize, get_nested

RATE_LIMIT_ERRORS = (ResourceExhausted, TooManyRequests)

# ---- Readable headers for the Excel output -----------------------------------------------
# Raw dotted field keys (e.g. "daily_function.use_telephone") are the right internal
# identifier but a bad column header -- reviewers work from the numbered paper form, not the
# key name. Each header below is "<item number>. <short label>" so it lines up with the form,
# with the full question text attached as a cell comment (hover to see it) rather than crammed
# into the header itself.
_KEY_TO_ITEM_NUM = {key: num for num, key in FIELD_NUM_TO_KEY.items()}
_SPECIAL_HEADERS = {
    "source_file": "Form",
    "low_confidence_fields": "Self-Flagged Fields",
    "needs_review_fields": "Needs Review Fields",
}


def _short_label(field_key: str) -> str:
    if field_key in DAILY_FUNCTION_LABELS:
        return DAILY_FUNCTION_LABELS[field_key]
    last_segment = field_key.split(".")[-1]
    return last_segment.replace("_", " ").title()


def header_label(field_key: str) -> str:
    if field_key in _SPECIAL_HEADERS:
        return _SPECIAL_HEADERS[field_key]
    num = _KEY_TO_ITEM_NUM.get(field_key)
    label = _short_label(field_key)
    return f"{num}. {label}" if num is not None else label


def header_tooltip(field_key: str) -> Optional[str]:
    return QUESTION_TEXT.get(field_key) or DAILY_FUNCTION_LABELS.get(field_key)


def extract_one_form_with_retry(gemini_client, model, pdf_path, schema, max_attempts=5):
    """Wraps extract_one_form() with exponential backoff on 429/rate-limit errors specifically
    -- other errors (bad PDF, schema mismatch, etc.) still fail immediately rather than retrying
    something that will never succeed."""
    for attempt in range(1, max_attempts + 1):
        try:
            return extract_one_form(gemini_client, model, pdf_path, schema)
        except RATE_LIMIT_ERRORS:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt  # 2, 4, 8, 16s
            time.sleep(wait)


def merge_and_flag(source_file: str, baseline_rec: dict, candidate_rec: dict) -> tuple:
    """Returns (final_record, review_rows). See module docstring: disagreement is the trigger,
    not either system's own confidence -- self-flags are recorded for context only."""
    final_record: dict = {"source_file": source_file}
    review_rows = []

    baseline_self_flagged = set(baseline_rec.get("low_confidence_fields", []))
    candidate_self_flagged = set(candidate_rec.get("low_confidence_fields", []))

    for field_key in ALL_FIELD_KEYS:
        b_val = get_nested(baseline_rec, field_key)
        c_val = get_nested(candidate_rec, field_key)

        if normalize(b_val) == normalize(c_val):
            set_nested(final_record, field_key, b_val)
            continue

        # Disagreement -- the trigger for review, independent of self-reported confidence.
        set_nested(final_record, field_key, None)
        review_rows.append({
            "source_file": source_file,
            "field_key": field_key,
            "section": "daily_function" if field_key in DAILY_FUNCTION_KEYS else field_key.split(".")[0],
            "cloud_vision_value": b_val,
            "gemini_value": c_val,
            "cloud_vision_self_flagged_low_confidence": field_key in baseline_self_flagged,
            "gemini_self_flagged_uncertain": field_key in candidate_self_flagged,
            # This is the case the whole pipeline exists to catch: neither system raised its
            # own hand, and they still don't agree. Only cross-method disagreement caught it.
            "caught_only_by_disagreement": field_key not in baseline_self_flagged
                                            and field_key not in candidate_self_flagged,
        })

    final_record["needs_review_fields"] = [r["field_key"] for r in review_rows]
    return final_record, review_rows


def _column_sort_key(field_key: str):
    """Orders columns to match the numbered paper form (1-58) instead of the dict-insertion
    order the internal pipeline happens to use -- e.g. "final_observations" (item 58) no longer
    lands in the middle of the sheet just because it's defined early in QUESTION_TEXT. Identity
    columns (source_file) come first, review-metadata columns (self-flagged/needs-review lists)
    come last."""
    if field_key == "source_file":
        return (0, 0)
    if field_key in ("low_confidence_fields", "needs_review_fields"):
        return (2, 0)
    return (1, _KEY_TO_ITEM_NUM.get(field_key, 999))


def _flat_records_to_df(records: list) -> Tuple[pd.DataFrame, list]:
    flat_records = [flatten_for_csv(r) for r in records]
    all_keys: list = []
    for r in flat_records:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort(key=_column_sort_key)
    df = pd.DataFrame(flat_records, columns=all_keys)
    return df.rename(columns={k: header_label(k) for k in all_keys}), all_keys


REVIEW_WORKLIST_RENAME = {
    "source_file": "Form",
    "field_key": "Field",
    "section": "Section",
    "cloud_vision_value": "Cloud Vision Answer",
    "gemini_value": "Gemini Answer",
    "cloud_vision_self_flagged_low_confidence": "CV Self-Flagged?",
    "gemini_self_flagged_uncertain": "Gemini Self-Flagged?",
    "caught_only_by_disagreement": "Priority Review",
}
REVIEW_WORKLIST_COLS = list(REVIEW_WORKLIST_RENAME.keys())


def _style_sheet(ws, header_tooltips: dict, freeze_after_col: str = "A"):
    """Shared formatting: bold+wrapped header row, frozen header (and the leftmost identifying
    column), an autofilter, and column widths sized to content instead of Excel's default ~9
    characters -- that default width is the main reason the raw output was hard to review."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.comments import Comment
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    header_align = Alignment(wrap_text=True, vertical="center", horizontal="center")
    body_align = Alignment(wrap_text=True, vertical="top")

    max_row = ws.max_row
    max_col = ws.max_column
    for col_idx in range(1, max_col + 1):
        header_cell = ws.cell(row=1, column=col_idx)
        header_cell.font = header_font
        header_cell.fill = header_fill
        header_cell.alignment = header_align
        tooltip = header_tooltips.get(header_cell.value)
        if tooltip:
            header_cell.comment = Comment(tooltip, "pipeline")

        # Width: fit the header, but cap so a long question/answer doesn't blow out the sheet --
        # wrap_text on body cells handles the overflow instead of a mile-wide column.
        col_letter = get_column_letter(col_idx)
        header_len = len(str(header_cell.value or ""))
        sample_lens = [len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(2, min(max_row, 30) + 1)]
        width = max([header_len] + sample_lens + [10])
        ws.column_dimensions[col_letter].width = min(max(width, 12), 45)

    for row in ws.iter_rows(min_row=2, max_row=max_row, max_col=max_col):
        for cell in row:
            cell.alignment = body_align

    ws.freeze_panes = f"{freeze_after_col}2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 32


def _highlight_priority_rows(ws, priority_col_idx: int):
    """On the Review Worklist tab specifically: color the rows where neither system self-flagged
    but they still disagree -- the exact 'confidently wrong, unflagged' case this pipeline
    exists to catch, and the highest-priority rows for a human to look at first."""
    from openpyxl.styles import PatternFill
    priority_fill = PatternFill("solid", fgColor="FFC7CE")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        cell = row[priority_col_idx - 1]
        if str(cell.value).strip().lower() in ("true", "yes"):
            for c in row:
                c.fill = priority_fill


def write_workbook(out_path: Path, baseline_records: list, candidate_records: list,
                    final_records: list, review_rows: list):
    """One .xlsx with four tabs, instead of four separate json/csv files -- open once, page
    through Baseline/Gemini/Final/Review Worklist with the sheet tabs at the bottom. Headers are
    "<item number>. <short label>" matching the numbered paper form, with the full question text
    attached as a hover comment; header row is bold/frozen; columns are sized to content."""
    baseline_df, baseline_keys = _flat_records_to_df(baseline_records)
    candidate_df, candidate_keys = _flat_records_to_df(candidate_records)
    final_df, final_keys = _flat_records_to_df(final_records)

    review_df = pd.DataFrame(review_rows, columns=REVIEW_WORKLIST_COLS)
    for bool_col in ("cloud_vision_self_flagged_low_confidence", "gemini_self_flagged_uncertain",
                     "caught_only_by_disagreement"):
        review_df[bool_col] = review_df[bool_col].map({True: "Yes", False: "No"})
    review_df["field_key"] = review_df["field_key"].map(lambda k: header_label(k))
    review_df = review_df.rename(columns=REVIEW_WORKLIST_RENAME)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
        candidate_df.to_excel(writer, sheet_name="Gemini", index=False)
        final_df.to_excel(writer, sheet_name="Final", index=False)
        review_df.to_excel(writer, sheet_name="Review Worklist", index=False)

        for sheet_name, keys in (("Baseline", baseline_keys), ("Gemini", candidate_keys), ("Final", final_keys)):
            tooltips = {header_label(k): header_tooltip(k) for k in keys}
            _style_sheet(writer.sheets[sheet_name], tooltips)

        _style_sheet(writer.sheets["Review Worklist"], {}, freeze_after_col="B")
        priority_col_idx = list(review_df.columns).index("Priority Review") + 1
        _highlight_priority_rows(writer.sheets["Review Worklist"], priority_col_idx)


def write_json_output(out_path: Path, baseline_records: list, candidate_records: list,
                       final_records: list, review_rows: list):
    """One combined JSON alongside the .xlsx -- same four groupings (baseline/gemini/final/
    review_worklist), but as nested records (unflattened dotted keys, real null/bool types)
    rather than Excel's flattened, header-relabeled, string-ified form. Meant for anyone who
    wants to script against the output (re-run comparisons, feed it into another system, etc.)
    rather than review it by eye."""
    combined = {
        "baseline": baseline_records,
        "gemini": candidate_records,
        "final": final_records,
        "review_worklist": review_rows,
    }
    out_path.write_text(json.dumps(combined, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--vision-schema", default=str(Path(__file__).parent / "home_assessment_schema.json"),
                     help="Calibrated column-layout schema for extract_assessment.py")
    ap.add_argument("--gemini-location", default="us-central1")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash",
                     help="gemini-2.5-flash is the default: pilot testing showed 100% agreement "
                          "with the Cloud Vision baseline at a fraction of gemini-2.5-pro's "
                          "latency. Pass gemini-2.5-pro to fall back to the slower/pricier model.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files (pilot batch)")
    ap.add_argument("--workers", type=int, default=5,
                     help="How many forms to process concurrently (each form still runs its own "
                          "Cloud Vision call then its own Gemini call, in that order). Default 5 "
                          "-- keep this modest to avoid tripping Vertex AI's per-project rate "
                          "limits; Gemini calls retry automatically on 429s regardless.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        sys.exit(f"No PDFs found in {input_dir}")

    vision_client = vision.ImageAnnotatorClient()
    vision_schema = json.loads(Path(args.vision_schema).read_text())

    gemini_client = genai.Client(vertexai=True, project=args.project_id, location=args.gemini_location)
    gemini_schema = build_flat_schema()

    def process_one(pdf_path: Path):
        """Runs on a worker thread: Cloud Vision, then Gemini, then merge -- for one form.
        Returns (pdf_path, baseline_rec, candidate_rec, final_rec, review_rows) or raises."""
        baseline_rec, _ = process_one_form_vision(vision_client, pdf_path, schema=vision_schema)
        candidate_rec = extract_one_form_with_retry(gemini_client, args.gemini_model, pdf_path, gemini_schema)
        final_rec, review_rows = merge_and_flag(pdf_path.name, baseline_rec, candidate_rec)
        return pdf_path, baseline_rec, candidate_rec, final_rec, review_rows

    baseline_records, candidate_records, final_records, all_review_rows = [], [], [], []
    failed = []

    print(f"Found {len(pdf_paths)} PDF(s). Running Cloud Vision OCR + Gemini "
          f"({args.workers} form(s) concurrently), then comparing...\n")

    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, p): p for p in pdf_paths}
        for future in as_completed(futures):
            pdf_path = futures[future]
            done_count += 1
            try:
                _, baseline_rec, candidate_rec, final_rec, review_rows = future.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{done_count}/{len(pdf_paths)}] {pdf_path.name} ... FAILED ({e})")
                failed.append((pdf_path.name, str(e)))
                continue

            silent_count = sum(1 for r in review_rows if r["caught_only_by_disagreement"])
            print(f"[{done_count}/{len(pdf_paths)}] {pdf_path.name} ... ok "
                  f"({len(review_rows)} field(s) disagree, {silent_count} not self-flagged by either system)")

            baseline_records.append(baseline_rec)
            candidate_records.append(candidate_rec)
            final_records.append(final_rec)
            all_review_rows.extend(review_rows)

    # Keep output rows in a stable, deterministic order regardless of which worker finished first.
    order = {p.name: i for i, p in enumerate(pdf_paths)}
    baseline_records.sort(key=lambda r: order[r["source_file"]])
    candidate_records.sort(key=lambda r: order[r["source_file"]])
    final_records.sort(key=lambda r: order[r["source_file"]])
    all_review_rows.sort(key=lambda r: (order[r["source_file"]], r["field_key"]))

    if failed:
        print(f"\n{len(failed)} form(s) failed and were skipped: {[f[0] for f in failed]}")

    xlsx_path = Path(f"{args.out_prefix}.xlsx")
    write_workbook(xlsx_path, baseline_records, candidate_records, final_records, all_review_rows)

    json_path = Path(f"{args.out_prefix}.json")
    write_json_output(json_path, baseline_records, candidate_records, final_records, all_review_rows)

    total_fields = len(final_records) * len(ALL_FIELD_KEYS)
    total_disagreements = len(all_review_rows)
    total_silent = sum(1 for r in all_review_rows if r["caught_only_by_disagreement"])

    print(f"\nDone. {len(final_records)} form(s) processed.")
    print(f"Overall field agreement: {total_fields - total_disagreements}/{total_fields} "
          f"({100 * (total_fields - total_disagreements) / total_fields:.1f}%)")
    print(f"Fields flagged for review: {total_disagreements}")
    print(f"  -- of those, {total_silent} were NOT self-flagged as low-confidence by either "
          f"system (this is the count that matters: these would have silently gone through "
          f"as 'confident' under a self-confidence-only approach, and were only caught here "
          f"because the two methods disagreed).")
    print(f"\nWrote {xlsx_path} (tabs: Baseline, Gemini, Final, Review Worklist -- start review "
          f"on the Review Worklist tab, sorted by source_file; 'caught_only_by_disagreement' "
          f"marks the highest-priority rows).")
    print(f"Wrote {json_path} (same four groupings as nested records -- baseline/gemini/final/"
          f"review_worklist -- for scripting against rather than eyeballing).")


if __name__ == "__main__":
    main()
