import re, json, pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
FILE_PATH   = "/Users/prabhleenkaur/Code/GeneDrugChat/Data/tier1_smiles.csv"
COLUMN_NAME = "Interaction"
OUT_DIR     = Path("./cls_data")
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV     = OUT_DIR / "segments_2head.csv"
OUT_JSONL   = OUT_DIR / "segments_2head.jsonl"
VOCAB_JSON  = OUT_DIR / "vocab_2head.json"

DIR_INCREASE = ["increased"]
DIR_DECREASE = ["decreased"]

TARGET_GROUPS = {
    "chemical_modification": [
        r"\breduction\b", r"\bhydrolysis\b", r"\bhydroxylation\b",
        r"\bglutathionylation\b", r"\balkylation\b", r"\boxidation\b",
        r"\bmutagenesis\b", r"\badp-?ribosylation\b", r"\bsumoylation\b",
        r"\bacetylation\b",  r"\bubiquitination\b", r"\blipidation\b", r"\bsulfation\b",
        r"\bglucuronidation\b", r"\bnitrosation\b", r"\bcarbamoylation\b"
    ],
    "folding": [r"\bfolding\b"],
    "susceptibility": [r"\bsusceptibility\b"],
    "activity": [r"\bactivity\b"],
    "localization": [r"\blocalization\b", r"\bexport\b", r"\bimport\b",
                     r"\buptake\b", r"\btransport\b", r"\bsecretion\b"],
    "expression": [r"\bexpression\b"],
    "phosphorylation": [r"\bphosphorylation\b"],
    "metabolism": [r"\bmetabolism\b"],
    "abundance": [r"\babundance\b"],
    "chemical synthesis": [r"\bchemical synthesis\b"],
    "stability_degradation": [r"\bdegradation\b", r"\bstability\b", r"\bcleavage\b"],
    "methylation": [r"\bmethylation\b"],
    "splicing": [r"\bsplicing\b"],
    "binds_to": [r"\bbinds to\b"]
}

# ---------- REGEX ----------
INC_RE     = re.compile("|".join(DIR_INCREASE), re.IGNORECASE)
DEC_RE     = re.compile("|".join(DIR_DECREASE), re.IGNORECASE)
TARGET_RES = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in TARGET_GROUPS.items()}

TAG_RE     = re.compile(r"\[(?:DRUG|PROTEIN)[^\]]*\]", re.IGNORECASE)
AND_SPLIT  = re.compile(r"\s+and\s+", re.IGNORECASE)
NON_ALPHA  = re.compile(r"[^a-z\s]")

# ---------- HELPERS ----------
def normalize(text: str) -> str:
    text = text.lower()
    text = NON_ALPHA.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def pick_direction(seg_norm: str) -> str:
    if INC_RE.search(seg_norm): return "increase"
    if DEC_RE.search(seg_norm): return "decrease"
    return "none"

def pick_target(seg_norm: str) -> str:
    for name, rx in TARGET_RES.items():
        if rx.search(seg_norm):
            return name
    return "other_prop"

# ---------- MAIN ----------
df = pd.read_csv(FILE_PATH)

rows = []
for row_idx, row in df.iterrows():
    txt = row.get(COLUMN_NAME, "")
    if not isinstance(txt, str) or not txt.strip():
        continue
    gene_id = row.get("GeneID", "")
    smiles  = row.get("SMILES", "")
    segments = AND_SPLIT.split(txt)

    seg_id = 0
    for seg in segments:
        seg_clean = TAG_RE.sub("", seg).strip()
        seg_norm  = normalize(seg_clean)
        if not seg_norm:
            continue
        target    = pick_target(seg_norm)
        direction = pick_direction(seg_norm)
        rows.append({
            "row_idx": row_idx,
            "seg_idx": seg_id,
            "gene_id": gene_id,
            "smiles": smiles,
            "segment_text": seg_clean,
            "target": target,
            "direction": direction
        })
        seg_id += 1

# Build vocabularies
targets    = sorted({r["target"] for r in rows})
directions = sorted({r["direction"] for r in rows})
target2id  = {t: i for i, t in enumerate(targets)}
direction2id = {d: i for i, d in enumerate(directions)}

# Add IDs
for r in rows:
    r["target_id"]    = target2id[r["target"]]
    r["direction_id"] = direction2id[r["direction"]]

from collections import Counter

target_counts = Counter(r["target"] for r in rows)
direction_counts = Counter(r["direction"] for r in rows)

print("\n[Target Class Counts]")
for k, v in target_counts.items():
    print(f"{k:>20}: {v}")

print("\n[Direction Class Counts]")
for k, v in direction_counts.items():
    print(f"{k:>20}: {v}")

# Save CSV
seg_df = pd.DataFrame(rows)
seg_df.to_csv(OUT_CSV, index=False)

# Save JSONL
with open(OUT_JSONL, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# Save vocab maps
with open(VOCAB_JSON, "w") as f:
    json.dump({"target2id": target2id, "direction2id": direction2id}, f, indent=2)

print(f"Segments written: {len(rows)}")
print(f"CSV:    {OUT_CSV}")
print(f"JSONL:  {OUT_JSONL}")
print(f"Vocabs: {VOCAB_JSON}")
print("Targets:", target2id)
print("Directions:", direction2id)