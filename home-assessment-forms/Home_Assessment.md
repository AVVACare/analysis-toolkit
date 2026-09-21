# Home Assessment Extraction Pipeline: Technical Solution

## 1. Objective

Build a reliable extraction pipeline that converts roughly 100s of  scanned In-Home Assessment Documentation Form PDFs into structured records, covering header/identity fields, behavioral and environmental/social questions, and the 15-item Daily Function rating table. Success criteria: field-level accuracy high enough to trust automated extraction for most fields, plus a defensible, auditable process for routing the remainder to human review instead of silently accepting a bad read.

## 2. Solution Architecture

Three extraction backends were evaluated. Google Cloud Document AI's Form Parser was ruled out first: its generic table-structure model collapses the Daily Function section's 5 rating columns into a 4-column template, losing which column was marked.

The architecture that shipped is a dual-model cross-validation design: a Cloud Vision OCR extractor (`extract_home_assessment_forms_cloud_vision.py`) reconstructs fields geometrically -- gridline detection for row boundaries, ink-density comparison for which rating column is marked, digit-vs-checkmark classification -- and a Gemini extractor (`extract_home_assessment_forms_gemini.py`) reads the whole scanned page natively in a single multimodal call against a fixed JSON schema (`build_flat_schema()`), covering the same field keys as the Cloud Vision output. A merge layer (`merge_and_flag()` in `run_ocr_llm_pipeline.py`) reconciles the two outputs field by field, so no single model's output reaches the final record unchecked.

## 3. Processing Pipeline & Concurrency Model

Per form: run the Cloud Vision extractor, then the Gemini extractor, then `merge_and_flag()` to reconcile. `run_ocr_llm_pipeline.py` runs this per-form pipeline across a `ThreadPoolExecutor` (`--workers`, default 5), so up to 5 forms are in flight concurrently -- each doing its own Cloud Vision call followed by its own Gemini call, not perfectly synchronized across workers.

The worker count is capped deliberately: Vertex AI's Gemini endpoints enforce per-project rate limits, and higher concurrency risks 429 (rate-limit) responses. `extract_one_form_with_retry()` wraps each Gemini call with exponential backoff (2, 4, 8, 16s, up to 5 attempts) specifically on 429/ResourceExhausted errors -- other failures (bad PDF, schema mismatch) fail immediately rather than retrying something that can't succeed. Output records are re-sorted back into input file order after the pool completes, since completion order depends on which forms finish first.

## 4. Comparison & Merge Logic

`merge_and_flag()` compares every field key between the two extractors using a normalization step (`normalize()`) that treats None, empty string, and whitespace-only as an equivalent "no answer", and compares strings case- and whitespace-insensitively -- so a formatting difference doesn't register as a false disagreement.

Where the two agree, that value is trusted and written to the final record. Where they disagree, the final record gets null for that field (no guess at which system is right) and a row is added to the review worklist with both raw values, each system's own self-reported confidence flag for that field, and a `caught_only_by_disagreement` flag -- true when neither system flagged its own answer as uncertain, meaning cross-method disagreement was the only thing that caught it. That is the core design bet of this pipeline: self-reported OCR/LLM confidence is not well-calibrated enough to trust alone, so an independent second system with a different failure mode is the check, not either model's own introspection.

## 5. Output Artifacts

Each pipeline run produces one Excel workbook (plus a matching JSON for scripting) with four tabs:

- Baseline -- raw Cloud Vision output, one row per form.
- Gemini -- raw Gemini output, one row per form.
- Final -- the merged record per form: fields both systems agreed on are filled in, disagreements are left blank.
- Review Worklist -- one row per disagreeing field across every form, with both systems' raw values side by side.

Start on the Review Worklist tab, sorted by form. Rows highlighted pink are the highest priority: fields where neither system flagged its own answer as low-confidence, but the two still disagreed -- exactly the "confidently wrong" case that a single system's self-reported confidence would have missed.

## 6. Validation Methodology & Results

To validate the merge logic's output against ground truth -- not just cross-model agreement -- we drew a stratified 30-file sample from the 60-file pilot batch: the 20 files with the most Cloud Vision/Gemini disagreements (worst-case coverage) plus 10 randomly selected files (baseline coverage), then manually verified every flagged field against the original scan. That produced 754 verified rows across 38 fields.

Daily Function is the weakest section by a wide margin: most of its 15 rating columns matched ground truth only 65-90% of the time, regardless of where the review threshold is set, so the section fails as a whole rather than field-by-field. The safety-relevant Yes/No fields (falls, exploitation, housing/food insecurity, social isolation) held up well at 94%+ match rate, staying clear of mandatory review at any reasonable threshold. Two identity fields stood out as outliers independent of section: Medicare Identification Number matched only 62% of the time and Patient Name 83%, both below what their downstream consequence (billing and record-matching accuracy) can tolerate.

## 7. Design Decisions, Limitations & Next Steps

### Design Decisions

Promote Gemini to the primary, trusted extractor in the production pipeline. Keep Cloud Vision as an independent secondary check: where the two disagree, log the field in a review worklist for visibility rather than blanking it out in the final record. This preserves an audit trail and still catches genuine Gemini misses, without discarding correct Gemini answers just because Cloud Vision had a bad read on a difficult scan -- the cost of the prior null-on-disagreement policy.

### Limitations

Spot-check sample sizes are modest (15-25 rows per field), so field-level match rates should be read as directional rather than statistically precise. Cloud Vision's extractor depends on calibrated geometric anchors and degrades unpredictably on skewed, low-resolution, or low-contrast scans -- a failure mode it cannot self-detect. Neither system's self-reported confidence is well-calibrated enough to trust alone, which is the whole reason the merge logic relies on cross-method disagreement instead. Gemini's own accuracy gap on Daily Function has not yet been root-caused beyond "a 5-way visual judgment is hard to make from a single whole-page read" -- see Next Steps for planned mitigations.

### Next Steps

1. Update the pipeline's merge logic (`merge_and_flag` in `run_ocr_llm_pipeline.py`) to adopt Gemini's value by default on disagreement, keeping Cloud Vision's value and the disagreement flag as review context.
2. Continue running the remaining forms in the batch.
3. Monitor the Review Worklist tab per file -- an unusually high disagreement count for a single file is a useful proxy for a rough scan worth a manual look, independent of either system's self-reported confidence.
4. Lock in the per-field review policy: an 85% general match-rate threshold, mandatory review for the entire Daily Function section regardless of measured rate, and mandatory review for Medicare ID and Patient Name regardless of threshold.
5. Explore ways to improve Gemini's Daily Function accuracy specifically -- crop and zoom the rating-table region before sending it, use gemini-2.5-pro for that section, and add a self-consistency check across repeated calls.
