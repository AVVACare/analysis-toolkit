#!/usr/bin/env python3
"""
Flatten + profile the home-assessment extraction output (home_assessment.json) before doing any
substantive analytics on it. This is step 1 of the analytics plan: know what you actually have
(null rates, low-confidence-field frequency, value distributions) before drawing conclusions from
any single field.

No DSS/CaReS scoring here on purpose -- this is categorization/profiling only, per the current
scope decision. That integration can come later once the derived signals below are settled.

Usage:
    python profile_home_assessment.py --input home_assessment.json --out-prefix ./profile

Outputs:
    {out-prefix}_field_profile.csv   -- one row per field: type, null rate, low-confidence rate,
                                         and (for boolean/choice fields) its value distribution
    {out-prefix}_flat.csv            -- one row per patient, one column per field (for pivoting /
                                         further analysis in Excel or pandas)
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Fields that are identifiers/free text/dates -- profiled for null-rate and low-confidence rate
# only, not value-distribution (a distribution over 100 distinct names or addresses isn't useful).
FREE_TEXT_OR_ID_FIELDS = {
    "header.address", "header.answers_provided_by", "header.date_of_birth",
    "header.in_home_assessment_date", "header.living_arrangement_details",
    "header.medicare_id", "header.patient_name", "header.performed_by",
    "header.referral_date", "home_safety.things_missing_that_could_help",
    "behavior.other_observations", "environmental_social.other_observations",
    "final_observations",
}


def flatten(rec: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in rec.items():
        if k in ("source_file", "low_confidence_fields"):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="home_assessment.json from the Gemini extraction pipeline")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    records = json.loads(Path(args.input).read_text())
    n = len(records)
    if not n:
        raise SystemExit("No records found in input.")

    # Discover every field key in first-seen order, across all records (a record missing a
    # section entirely still needs its fields counted as null, not skipped).
    all_keys: list = []
    seen = set()
    flat_records = []
    for r in records:
        flat = flatten(r)
        flat["source_file"] = r.get("source_file")
        flat_records.append(flat)
        for k in flat:
            if k not in seen and k != "source_file":
                seen.add(k)
                all_keys.append(k)

    low_conf_counter = Counter()
    for r in records:
        for k in (r.get("low_confidence_fields") or []):
            low_conf_counter[k] += 1

    profile_rows = []
    for key in all_keys:
        values = [fr.get(key) for fr in flat_records]
        non_null = [v for v in values if v is not None and v != ""]
        null_count = n - len(non_null)
        low_conf_count = low_conf_counter.get(key, 0)

        if key in FREE_TEXT_OR_ID_FIELDS:
            field_type = "text"
            distribution = ""
        elif all(isinstance(v, bool) or v is None for v in values):
            field_type = "boolean"
            dist = Counter(v for v in values if v is not None)
            distribution = f"True={dist.get(True, 0)}, False={dist.get(False, 0)}"
        else:
            field_type = "choice"
            dist = Counter(non_null)
            distribution = ", ".join(f"{k}={v}" for k, v in dist.most_common())

        profile_rows.append({
            "field": key,
            "type": field_type,
            "null_count": null_count,
            "null_rate_pct": round(100 * null_count / n, 1),
            "low_confidence_count": low_conf_count,
            "low_confidence_rate_pct": round(100 * low_conf_count / n, 1),
            "value_distribution": distribution,
        })

    # Surface the fields most worth a second look first: high null rate or high low-confidence
    # rate suggests either a systemically hard-to-read question or a form section that's often
    # left blank -- either way, worth knowing before trusting prevalence numbers from that field.
    profile_rows.sort(key=lambda r: (r["low_confidence_rate_pct"], r["null_rate_pct"]), reverse=True)

    out_profile = Path(f"{args.out_prefix}_field_profile.csv")
    with out_profile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    out_flat = Path(f"{args.out_prefix}_flat.csv")
    fieldnames = ["source_file"] + all_keys + ["low_confidence_field_count"]
    with out_flat.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, fr in zip(records, flat_records):
            row = dict(fr)
            row["low_confidence_field_count"] = len(r.get("low_confidence_fields") or [])
            writer.writerow(row)

    print(f"{n} records profiled across {len(all_keys)} fields.")
    print(f"Wrote {out_profile} and {out_flat}.")
    print("\nTop 10 fields by low-confidence rate (worth a second look before trusting):")
    for row in profile_rows[:10]:
        print(f"  {row['field']:55s} low_conf={row['low_confidence_rate_pct']:5.1f}%  "
              f"null={row['null_rate_pct']:5.1f}%  {row['value_distribution']}")


if __name__ == "__main__":
    main()
