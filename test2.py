# test_parser.py  — run this first, takes 2 seconds
import re
from pathlib import Path

PREFIX_TO_CODE = {
    "backpack"      : "bg",
    "normal"        : "nm",
    "hand_in_pocket": "hp",
    "gown"          : "cl",
    "static_occ"    : "so",
    "dynamic_occ"   : "do",
}

def parse_filename(fname):
    stem  = Path(fname).stem
    parts = stem.split("_id")
    if len(parts) != 2:
        return None
    prefix = parts[0].lower()
    rest   = parts[1]
    m = re.match(r'^(\d+)_(\d+)$', rest)
    if not m:
        return None
    subj_id = int(m.group(1))
    trial   = int(m.group(2))
    cond    = PREFIX_TO_CODE.get(prefix)
    if cond is None:
        print(f"  WARNING unknown prefix: '{prefix}'")
        return None
    return cond, subj_id, trial

# Test all 6 patterns
test_files = [
    "backpack_id023_3.avi",
    "normal_id023_1.avi",
    "hand_in_pocket_id023_2.avi",
    "gown_id023_1.avi",
    "static_occ_id023_1.avi",
    "dynamic_occ_id023_1.avi",
]

for f in test_files:
    result = parse_filename(f)
    print(f"{f:35s}  →  {result}")