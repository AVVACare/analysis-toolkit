#!/usr/bin/env python3
"""
extract_assessment.py
----------------------
Converts a scanned/handwritten "In-Home Assessment Documentation Form" PDF
into structured data (Field #, Field, Response) using Google Cloud Vision
OCR (DOCUMENT_TEXT_DETECTION) - no LLM API required.

How it works
------------
1. Each PDF page is rendered to an image.
2. The image is sent to Google Cloud Vision's DOCUMENT_TEXT_DETECTION,
   which returns every recognized word with its pixel bounding box.
3. For "standard" fields (free-text/Y-N answers), the field number is
   located in the left-hand "Field #" column; every word that falls in the
   same row and in the right-hand response area is stitched together
   (reading order) into that field's "Response".
4. For "rating" fields (the Daily Function table, which is answered with a
   checkmark rather than text), OCR can't see a checkmark. Instead, the
   script crops each of the 5 rating-column cells for that row directly
   from the image and measures ink-pixel density - the column with
   meaningfully more dark pixels than the others is taken as the checked
   answer. Rows where no column stands out are flagged for manual review.

This is OCR + simple image analysis, not a reasoning model - it will not be
as robust as an LLM-vision approach on messy handwriting. Always spot-check
the "needs_review" rows in the output.

Usage
-----
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    pip install google-cloud-vision pdf2image pandas pillow numpy
    python3 extract_assessment.py input.pdf schema.json output_name

Outputs: output_name.json, output_name.csv

Also importable: extract_rows_for_pdf(pdf_path, schema, client, img_dir) ->
list of {field_num, field, response, needs_review} rows, in schema order --
used by extract_home_assessment_forms_cloud_vision.py to plug this into the
rest of the pipeline (field-key mapping, Y/N and rating normalization,
comparison against the Gemini leg) without touching any logic below.
"""

import sys
import os
import json
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from google.cloud import vision


# --------------------------------------------------------------------------
# Gridline detection (independent of OCR) - used to get precise, reliable
# row boundaries rather than relying solely on where OCR happened to place
# a field-number word vertically within its row.
# --------------------------------------------------------------------------
def find_gridlines(image_path, x0_frac=0.03, x1_frac=0.97, dark_thresh=180, row_frac=0.5, thick_thresh=8):
    """Detect horizontal row-boundary lines.

    Thin lines (ordinary table borders, a few px tall) are represented by
    their average y. Thick dark bands (e.g. a shaded header row, tens of px
    tall) contribute BOTH their top and bottom edge as separate gridlines -
    a single row can need to snap to either edge depending on whether it's
    the row ending right before the band (needs the band's TOP edge) or the
    row starting right after it (needs the band's BOTTOM edge). Collapsing a
    thick band to just one edge silently breaks whichever row needed the
    other one - which is what caused header text to leak into the row
    immediately above a second table's header (e.g. a section boundary
    between a Y/N table and a following ratings table).
    """
    im = Image.open(image_path).convert("L")
    w, h = im.size
    arr = np.array(im)
    x0, x1 = int(x0_frac * w), int(x1_frac * w)
    sub = arr[:, x0:x1]
    row_dark_frac = (sub < dark_thresh).mean(axis=1)
    lines = [y for y in range(h) if row_dark_frac[y] > row_frac]
    if not lines:
        return []
    clusters = []
    cur = [lines[0]]
    for y in lines[1:]:
        if y - cur[-1] <= 3:
            cur.append(y)
        else:
            if (max(cur) - min(cur)) > thick_thresh:
                clusters.extend([min(cur), max(cur)])
            else:
                clusters.append(sum(cur) // len(cur))
            cur = [y]
    if (max(cur) - min(cur)) > thick_thresh:
        clusters.extend([min(cur), max(cur)])
    else:
        clusters.append(sum(cur) // len(cur))
    return sorted(set(clusters))


def snap_to_gridlines(y_center, gridlines, fallback_top, fallback_bottom):
    """Find the gridline pair straddling y_center; fall back to given bounds if none found."""
    above = [g for g in gridlines if g <= y_center]
    below = [g for g in gridlines if g >= y_center]
    top = max(above) if above else fallback_top
    bottom = min(below) if below else fallback_bottom
    if bottom <= top:
        return fallback_top, fallback_bottom
    return top, bottom


# --------------------------------------------------------------------------
# PDF -> images
# --------------------------------------------------------------------------
def pdf_to_images(pdf_path, out_dir, dpi=200):
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, page in enumerate(pages, start=1):
        p = os.path.join(out_dir, f"page-{i}.png")
        page.save(p, "PNG")
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# Google Cloud Vision OCR (service-account auth via the official client
# library - reads credentials automatically from GOOGLE_APPLICATION_CREDENTIALS)
# --------------------------------------------------------------------------
_vision_client = None


def get_vision_client():
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def ocr_page(image_path, client):
    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    if not annotation.pages:
        return [], 0, 0

    page_info = annotation.pages[0]
    page_w, page_h = page_info.width, page_info.height

    words = []
    order_index = 0
    for block in page_info.blocks:
        for para in block.paragraphs:
            for word in para.words:
                text = "".join(s.text for s in word.symbols)
                verts = word.bounding_box.vertices
                xs = [v.x for v in verts]
                ys = [v.y for v in verts]
                words.append(
                    {
                        "text": text,
                        "x_center": sum(xs) / len(xs),
                        "y_center": sum(ys) / len(ys),
                        "y_top": min(ys),
                        "y_bottom": max(ys),
                        "x_left": min(xs),
                        "x_right": max(xs),
                        # Position of this word in Vision's own traversal
                        # order (block -> paragraph -> word). Vision's
                        # DOCUMENT_TEXT_DETECTION groups text into blocks and
                        # paragraphs specifically to capture natural reading
                        # order, which is far more reliable for messy
                        # handwriting than re-deriving order from raw x/y
                        # coordinates ourselves.
                        "order_index": order_index,
                    }
                )
                order_index += 1
    return words, page_w, page_h


# --------------------------------------------------------------------------
# Row detection: find each field number in the left column, in order
# --------------------------------------------------------------------------
def find_field_anchors(words, page_w, page_h, layout, expected_nums):
    """Return {field_num: y_center_px} for field-number labels found on this page.

    A printed two-digit field number (e.g. "35") can occasionally get OCR'd
    as two separate single-digit words ("3", "5") rather than one - tight
    kerning on a small printed number, or reduced contrast against a shaded
    row background, can do this. If that number's anchor is never found, its
    entire row gets folded into whichever neighboring field number WAS
    found, which then shows up as several rows' worth of answers
    concatenated together under the wrong field, while the swallowed field
    numbers show up empty. So beyond matching single words directly, this
    also tries merging two horizontally-adjacent single-digit words on the
    same line into a two-digit number and checking that against the
    expected field list.
    """
    col_lo, col_hi = layout["field_num_col"]
    lo_px, hi_px = col_lo * page_w, col_hi * page_w
    expected_set = set(expected_nums)

    def clean(text):
        return "".join(ch for ch in text if ch.isdigit())

    candidates = [
        w for w in words
        if lo_px <= w["x_center"] <= hi_px and clean(w["text"])
    ]
    candidates.sort(key=lambda w: (w["y_center"], w["x_left"]))

    anchors = {}
    used = set()

    # Pass 1: direct single-word matches (the common case).
    for idx, w in enumerate(candidates):
        n_str = clean(w["text"])
        if n_str.isdigit() and int(n_str) in expected_set and int(n_str) not in anchors:
            anchors[int(n_str)] = w["y_center"]
            used.add(idx)

    # Pass 2: merge adjacent single-digit words on the same line into a
    # two-digit number (handles the split-number case above).
    for idx in range(len(candidates) - 1):
        if idx in used or (idx + 1) in used:
            continue
        w1, w2 = candidates[idx], candidates[idx + 1]
        same_line = abs(w1["y_center"] - w2["y_center"]) < 15
        adjacent = 0 <= (w2["x_left"] - w1["x_right"]) < 25
        if not (same_line and adjacent):
            continue
        combined = clean(w1["text"]) + clean(w2["text"])
        if combined.isdigit() and int(combined) in expected_set and int(combined) not in anchors:
            anchors[int(combined)] = (w1["y_center"] + w2["y_center"]) / 2
            used.add(idx)
            used.add(idx + 1)

    return anchors


def assemble_standard_response(words, page_w, y_top, y_bottom, layout):
    """Read the full response area (Answer column + Comments column together).

    Rather than re-deriving reading order from raw word coordinates (fragile
    against handwriting slant, uneven line spacing, and short answers that
    don't align neatly with comment lines), this filters to the words that
    fall in the target row/column region and then sorts them by Vision's own
    traversal order (order_index) - the order it grouped them into blocks and
    paragraphs, which is its own internal determination of reading order and
    is generally more reliable than a hand-rolled geometric heuristic."""
    std = layout["standard_table"]
    lo_px = std["answer_col"][0] * page_w
    hi_px = std["comment_col"][1] * page_w

    in_area = [
        w for w in words
        if y_top <= w["y_center"] < y_bottom and lo_px <= w["x_center"] <= hi_px
    ]
    if not in_area:
        return ""

    in_area.sort(key=lambda w: w["order_index"])
    return " ".join(w["text"] for w in in_area).strip()


# --------------------------------------------------------------------------
# Ink-density checkmark detection (no OCR needed for this part)
# --------------------------------------------------------------------------
def detect_rating_from_ocr_digit(words, page_w, y_top, y_bottom, layout):
    """Some forms answer the rating table by writing the number itself (1-5)
    into the cell rather than a checkmark. Reading that digit directly via
    OCR is far more reliable than ink-density comparison for this style,
    because digits vary hugely in ink mass by design (a '1' is a single thin
    stroke, a '4' has multiple strokes/a loop) - so "whichever column has
    more ink" is the wrong signal when the correct answer might be the
    lightest digit. Returns (label, needs_review) or None if no digit found
    (i.e. this row is likely answered with a checkmark instead, and the
    caller should fall back to ink-density detection)."""
    lo_px = layout["rating_table"]["field_label_col"][1] * page_w
    hi_px = page_w  # to the right edge - digit could sit anywhere across the 5 rating columns
    candidates = [
        w for w in words
        if y_top <= w["y_center"] < y_bottom and lo_px <= w["x_center"] <= hi_px
        and w["text"].strip(".,") in {"1", "2", "3", "4", "5"}
    ]
    if not candidates:
        return None

    digits_found = {w["text"].strip(".,") for w in candidates}
    labels = layout["rating_table"]["rating_col_labels"]
    if len(digits_found) > 1:
        # Multiple digits in one row is a valid answer (e.g. the patient's
        # ability varies by task/context) - report all of them rather than
        # flagging for review.
        matched = [labels[int(d) - 1] for d in sorted(digits_found)]
        return "; ".join(matched), False

    digit = int(next(iter(digits_found)))
    return labels[digit - 1], False


def detect_checked_rating(image_path, page_w, page_h, y_top, y_bottom, layout):
    im = Image.open(image_path).convert("L")  # grayscale
    inset_y = max(2, int((y_bottom - y_top) * 0.15))
    top = int(y_top) + inset_y
    bottom = int(y_bottom) - inset_y
    if bottom <= top:
        top, bottom = int(y_top), int(y_bottom)

    densities = []
    for (lo, hi) in layout["rating_table"]["rating_cols"]:
        left = int(lo * page_w) + 4
        right = int(hi * page_w) - 4
        if right <= left:
            densities.append(0.0)
            continue
        crop = im.crop((left, top, right, bottom))
        pixels = list(crop.getdata())
        if not pixels:
            densities.append(0.0)
            continue
        dark = sum(1 for p in pixels if p < 140)
        densities.append(dark / len(pixels))

    if not densities:
        return "NEEDS REVIEW (no columns configured)", densities

    max_d = max(densities)
    sorted_d = sorted(densities, reverse=True)
    runner_up = sorted_d[1] if len(sorted_d) > 1 else 0.0

    # Real checkmarks vary a lot in weight - a thin ballpoint tick can be
    # ~0.01 density while a bold marker check can be ~0.03-0.05. Empty cells
    # aren't always exactly 0 either - crop edges/scan noise can put an
    # unmarked column at ~0.005-0.008. So the floor for "there's a mark here"
    # has to sit between those two ranges, and the real signal is the GAP
    # between the winning column and the rest, not an absolute density value.
    MIN_DENSITY = 0.009   # lowest density that counts as "a mark is here"
    MIN_MARGIN = 0.006    # winner must clearly beat the runner-up

    if max_d < MIN_DENSITY:
        return "NEEDS REVIEW (no mark detected)", densities
    if (max_d - runner_up) < MIN_MARGIN:
        return "NEEDS REVIEW (ambiguous - marks too close to call)", densities

    idx = densities.index(max_d)
    return layout["rating_table"]["rating_col_labels"][idx], densities


def find_column_divider_x(image_path, page_w, expected_frac, search_frac=0.03, y0_frac=0.05, y1_frac=0.9):
    """Locate the actual pixel x-position of a vertical column divider near
    an expected fraction of page width, by searching a small window around
    that fraction for the strongest vertical line (measured across a broad
    span of rows). Different PDF renderings/scans of "the same" form layout
    can be off by a percent or two from a fraction calibrated on another
    file, which is enough to miss a divider entirely with a narrow check -
    so this re-locates it per document rather than trusting the stored
    fraction exactly."""
    im = Image.open(image_path).convert("L")
    w, h = im.size
    arr = np.array(im)
    search_px = max(3, int(search_frac * w))
    center = int(expected_frac * w)
    y0, y1 = int(y0_frac * h), int(y1_frac * h)
    col_dark_frac = (arr[y0:y1, :] < 180).mean(axis=0)
    lo, hi = max(0, center - search_px), min(w, center + search_px)
    best_x = max(range(lo, hi), key=lambda x: col_dark_frac[x])
    return best_x


def is_field_row_band(image_path, page_w, top, bottom, divider_x, min_height=15, dark_thresh=180):
    """A real field row has a continuous vertical divider between the Field#
    and Field columns; a full-width section-title band ("Environmental /
    Social: Are there any concerns...") does not, since the title text spans
    across where that divider would be. Checking for that divider's presence
    is a simple, OCR-independent way to tell field rows apart from title
    bands, so title bands can be excluded from row numbering entirely."""
    if bottom - top < min_height:
        return False
    im = Image.open(image_path).convert("L")
    arr = np.array(im)
    margin = max(2, int((bottom - top) * 0.15))
    strip = arr[top + margin:bottom - margin, max(0, divider_x - 3):divider_x + 4]
    if strip.size == 0:
        return False
    return (strip < dark_thresh).mean() > 0.4


def recover_multiple_missed_borders(image_path, bands, target_count, weak_row_frac=0.08, min_piece_height=30):
    """Some table rules render lighter than others and can fall below the
    normal gridline threshold entirely - and a gap can be missing more than
    one border this way, not just one. Rather than only handling a deficit
    of exactly one (which leaves any larger gap completely unresolved),
    this repeatedly finds the single strongest remaining low-threshold split
    candidate across all current bands and applies it, until either the
    band count reaches the target or no further plausible candidate can be
    found (at which point the gap is left unresolved rather than guessed)."""
    bands = list(bands)
    if not bands:
        return bands

    im = Image.open(image_path).convert("L")
    w = im.size[0]
    arr = np.array(im)
    x0, x1 = int(0.03 * w), int(0.97 * w)

    while len(bands) < target_count:
        best = None  # (dark_frac, band_index, split_y)
        for i, (t, b) in enumerate(bands):
            inner_top, inner_bottom = t + min_piece_height, b - min_piece_height
            if inner_bottom <= inner_top:
                continue
            sub = arr[inner_top:inner_bottom, x0:x1]
            row_dark_frac = (sub < 180).mean(axis=1)
            offset = int(row_dark_frac.argmax())
            strength = row_dark_frac[offset]
            if strength > weak_row_frac and (best is None or strength > best[0]):
                best = (strength, i, inner_top + offset)
        if best is None:
            break
        _, i, split_y = best
        t, b = bands[i]
        bands[i:i + 1] = [(t, split_y), (split_y, b)]

    return bands


def find_field_rows(words, page_w, page_h, layout, expected_nums, image_path):
    """Determine each field's (top, bottom) row boundaries.

    Baseline: for every field number OCR successfully read, use its known
    working approach directly - gridline-snap around that anchor's position.
    This alone is what already worked correctly on pages where OCR reads
    every field number fine (e.g. a ratings-table page with its own header
    layout), so it's left untouched for those.

    Gap-filling: for any field number OCR did NOT find, but which sits
    between two field numbers OCR DID find, use the page's gridline bands
    (reliable, OCR-independent) restricted to just that narrow gap to
    recover the missing row(s) - filtering out non-field bands (a
    section-title row spanning the gap) via is_field_row_band, and using a
    targeted low-threshold search for a border faint enough to have been
    missed entirely if the gap is short by exactly one band. This is scoped
    to just the gap between two confirmed anchors, rather than re-deriving
    the whole page's numbering, so it can't disturb a page/section that
    already resolved correctly on its own.
    """
    raw_anchors = find_field_anchors(words, page_w, page_h, layout, expected_nums)
    if not raw_anchors:
        return {}

    gridlines = find_gridlines(image_path)
    result = {}
    sorted_nums = sorted(raw_anchors.keys())

    # Baseline: gridline-snap around each directly-found anchor (unchanged
    # from the approach that already worked).
    for i, num in enumerate(sorted_nums):
        y_center = raw_anchors[num]
        fallback_top = y_center
        fallback_bottom = raw_anchors[sorted_nums[i + 1]] if i + 1 < len(sorted_nums) else page_h
        result[num] = snap_to_gridlines(y_center, gridlines, fallback_top, fallback_bottom)

    # Gap-filling: for each pair of consecutively-found anchors with missing
    # numbers between them, recover those rows from gridline bands local to
    # just that gap.
    field_num_col_right = layout["field_num_col"][1]
    divider_x = None
    for i in range(len(sorted_nums) - 1):
        num1, num2 = sorted_nums[i], sorted_nums[i + 1]
        missing_count = num2 - num1 - 1
        if missing_count <= 0:
            continue

        y1, y2 = raw_anchors[num1], raw_anchors[num2]
        # Bands strictly between the two anchors' own rows: start just after
        # num1's row top (its snapped top edge already computed above) and
        # end at num2's row top.
        gap_top = result[num1][1]
        gap_bottom = result[num2][0]
        local_bands = [(t, b) for (t, b) in zip(gridlines[:-1], gridlines[1:]) if t >= gap_top and b <= gap_bottom]
        if not local_bands:
            continue

        if divider_x is None:
            divider_x = find_column_divider_x(image_path, page_w, field_num_col_right)
        field_local_bands = [
            (t, b) for (t, b) in local_bands
            if is_field_row_band(image_path, page_w, t, b, divider_x)
        ]

        if len(field_local_bands) < missing_count:
            field_local_bands = recover_multiple_missed_borders(
                image_path, field_local_bands, missing_count
            )

        if len(field_local_bands) != missing_count:
            continue  # couldn't reconcile - leave these fields unresolved

        for offset, (t, b) in enumerate(field_local_bands):
            result[num1 + 1 + offset] = (t, b)

    return {num: bounds for num, bounds in result.items() if num in expected_nums}


def has_ink_in_response_area(image_path, page_w, y_top, y_bottom, layout, dark_thresh=180, min_density=0.01):
    """Check whether there's any handwritten mark at all in a standard
    field's response area, independent of OCR. Used when OCR returns no
    text for a cell, to distinguish a genuinely blank cell from one where a
    mark exists but wasn't legible enough for OCR to transcribe - the two
    need different follow-up (nothing to check vs. a specific mark to read
    by eye)."""
    std = layout["standard_table"]
    left = int(std["answer_col"][0] * page_w)
    right = int(std["comment_col"][1] * page_w)
    im = Image.open(image_path).convert("L")
    arr = np.array(im)
    top, bottom = int(y_top), int(y_bottom)
    if bottom <= top or right <= left:
        return False
    crop = arr[top:bottom, left:right]
    if crop.size == 0:
        return False
    return (crop < dark_thresh).mean() > min_density


# --------------------------------------------------------------------------
# Handwriting-specific OCR corrections
# --------------------------------------------------------------------------
def correct_yn_misreads(field_def, text):
    """On some handwriting, a cursive 'N' gets OCR'd as the digit '2' - and
    the same stylized loop-and-tail shape can get read as a RUN of several
    '2's ('22', '222') rather than a clean single one, if the cursive stroke
    has multiple curves that each individually resemble a '2'. For a field
    that's known to be a Y/N question, a leading token of nothing but '2's
    can never be a real answer - so it's virtually certainly a misread 'N',
    correctable without needing a review flag."""
    if not text or "(Y/N)" not in field_def.get("label", ""):
        return text
    tokens = text.split(None, 1)
    if tokens and tokens[0].strip(".,") and set(tokens[0].strip(".,")) == {"2"}:
        rest = tokens[1] if len(tokens) > 1 else ""
        return ("N " + rest).strip()
    return text


# --------------------------------------------------------------------------
# Per-PDF extraction, importable for use by other pipeline scripts
# --------------------------------------------------------------------------
def extract_rows_for_pdf(pdf_path, schema, client, img_dir):
    """Runs the full extraction for one PDF and returns a list of
    {field_num, field, response, needs_review} dicts, in schema order --
    exactly what main() below writes to JSON/CSV, factored out so
    extract_home_assessment_forms_cloud_vision.py can call this directly
    instead of shelling out to the CLI."""
    layout = schema["layout"]
    fields = schema["fields"]
    all_nums = [f["num"] for f in fields]

    os.makedirs(img_dir, exist_ok=True)
    image_paths = pdf_to_images(pdf_path, img_dir)

    results = {}  # field_num -> {"response":..., "needs_review": bool}

    for img_path in image_paths:
        words, page_w, page_h = ocr_page(img_path, client)
        if not words:
            continue

        anchors = find_field_rows(words, page_w, page_h, layout, all_nums, img_path)
        if not anchors:
            continue

        sorted_nums = sorted(anchors.keys())
        for num in sorted_nums:
            y_top, y_bottom = anchors[num]
            field_def = next(f for f in fields if f["num"] == num)

            if field_def["table"] == "rating":
                digit_result = detect_rating_from_ocr_digit(words, page_w, y_top, y_bottom, layout)
                if digit_result is not None:
                    label, needs_review = digit_result
                    results[num] = {"response": label, "needs_review": needs_review}
                else:
                    label, densities = detect_checked_rating(
                        img_path, page_w, page_h, y_top, y_bottom, layout
                    )
                    results[num] = {
                        "response": label,
                        "needs_review": "NEEDS REVIEW" in label,
                    }
            else:
                text = assemble_standard_response(words, page_w, y_top, y_bottom, layout)
                text = correct_yn_misreads(field_def, text)
                if not text and has_ink_in_response_area(img_path, page_w, y_top, y_bottom, layout):
                    results[num] = {
                        "response": "NEEDS REVIEW (a mark is present but OCR could not read it - check the scan by eye)",
                        "needs_review": True,
                    }
                else:
                    results[num] = {
                        "response": text if text else None,
                        "needs_review": not text,
                    }

    rows = []
    for f in fields:
        r = results.get(f["num"], {"response": None, "needs_review": True})
        rows.append({
            "field_num": f["num"],
            "field": f["label"],
            "response": r["response"],
            "needs_review": r["needs_review"],
        })
    return rows


# --------------------------------------------------------------------------
# Main (CLI entry point)
# --------------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 extract_assessment.py <input.pdf> <schema.json> <output_prefix>")
        sys.exit(1)

    pdf_path, schema_path, out_prefix = sys.argv[1:4]

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set (or the file doesn't exist).\n"
            "Create a service account with Cloud Vision access, download its JSON key,\n"
            "then: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
        )
    client = get_vision_client()

    with open(schema_path) as f:
        schema = json.load(f)

    work_dir = os.path.dirname(os.path.abspath(out_prefix)) or "."
    img_dir = os.path.join(work_dir, "_pages_tmp")

    print("Rendering PDF pages to images and running OCR...")
    rows = extract_rows_for_pdf(pdf_path, schema, client, img_dir)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}.csv", index=False)
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(rows, f, indent=2)

    missing = df["needs_review"].sum()
    print(f"Done. Wrote {out_prefix}.json/.csv")
    print(f"{missing} of {len(df)} fields flagged for manual review.")


if __name__ == "__main__":
    main()
