"""Build the final labeled CPDB carcinogenicity dataset from the raw CPDBChemical.xls.

Produces carcinogenicity_final.csv: name, cas, carcinogenicity_label, smiles.

Label rule (per pdfs/ChemicalTableLegend.pdf, not guessed from raw symbols):
  - positive (1) if a numeric TD50 value appears for the compound in ANY species/sex
    group across the four sheets (Rats and Mice, Hamsters, Monkeys, Other Species) —
    "positive in any tested species" convention.
  - negative (0) if no species was positive, and at least one species/sex group has
    "-" (all tested experiments negative).
  - excluded if every species/sex group is "." (never tested) or "I" (only test was
    an NCI/NTP experiment evaluated as inadequate) — CPDB has no usable call.

CAS recovery: Excel/xlrd silently misparsed some CAS numbers matching a date-like
pattern (e.g. real CAS 7758-01-2) as literal dates (7758-01-02 00:00:00). Fixed by
keeping the 2-digit middle segment zero-padded and stripping only the artificial
zero added to the 1-digit checksum segment. Confirmed this affected 26 rows in the
Rats and Mice sheet alone — silently dropping real compounds if left unfixed.

SMILES resolution: CAS -> SMILES via PubChem PUG REST, rate-limited to match the
existing project convention (PUBCHEM_RATE_LIMIT_DELAY=0.2s, see app/core/config.py).
Compounds that don't resolve are almost entirely genuine mixtures, technical-grade
formulations, and natural-product extracts (Aroclor 1242, gum/oil/dextran mixtures)
with no single-molecule structure — not a resolution failure to chase further.
"""
import csv
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

XLS_PATH = "CPDBChemical.xls"
OUTPUT_PATH = "carcinogenicity_final.csv"
PUBCHEM_DELAY = 0.2

NUMERIC_RE = re.compile(r"^<?\d+(\.\d+)?([a-zA-Z](,[a-zA-Z])*)?$")
DATE_MANGLED_RE = re.compile(r"^(\d+)-(\d{2})-(\d{2}) 00:00:00$")
VALID_CAS_RE = re.compile(r"^\d+-\d{1,2}-\d$")


def classify_td50(val: str) -> str:
    s = str(val).strip()
    if s in (".", "nan", "None", ""):
        return "not_tested"
    if s == "I":
        return "inadequate"
    if len(s) == 1 and not s.isalnum():
        return "negative"
    if NUMERIC_RE.match(s):
        return "positive"
    return "unknown:" + repr(s)


def fix_cas(raw: str) -> str:
    s = str(raw).strip()
    m = DATE_MANGLED_RE.match(s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{mo}-{int(d)}"
    return s


def extract_labels() -> dict:
    label_map, name_map = {}, {}

    def add_call(cas_raw, name, lab):
        if lab is None:
            return
        cas = fix_cas(cas_raw)
        if not VALID_CAS_RE.match(cas):
            return
        if cas not in label_map:
            label_map[cas] = lab
            name_map[cas] = name
        elif lab == 1:
            label_map[cas] = 1  # positive-in-any-species upgrade

    rm = pd.read_excel(XLS_PATH, sheet_name="Rats and Mice", engine="xlrd", header=None, skiprows=2)
    rm.columns = ["name", "cas", "salmonella", "td50_rat", "td50_mouse", "rms", "rfs", "mms", "mfs"]
    for _, row in rm.iterrows():
        rat_c, mouse_c = classify_td50(row["td50_rat"]), classify_td50(row["td50_mouse"])
        lab = 1 if "positive" in (rat_c, mouse_c) else (0 if "negative" in (rat_c, mouse_c) else None)
        add_call(row["cas"], row["name"], lab)

    ham = pd.read_excel(XLS_PATH, sheet_name="Hamsters", engine="xlrd", header=None, skiprows=1)
    ham.columns = ["name", "cas", "salmonella", "td50", "ms", "fs"]
    for _, row in ham.iterrows():
        c = classify_td50(row["td50"])
        add_call(row["cas"], row["name"], 1 if c == "positive" else (0 if c == "negative" else None))

    mon = pd.read_excel(XLS_PATH, sheet_name="Monkeys", engine="xlrd", header=None, skiprows=2)
    mon.columns = ["name", "cas", "salmonella", "td50_rh", "td50_cy", "sr", "sc"]
    for _, row in mon.iterrows():
        rh, cy = classify_td50(row["td50_rh"]), classify_td50(row["td50_cy"])
        lab = 1 if "positive" in (rh, cy) else (0 if "negative" in (rh, cy) else None)
        add_call(row["cas"], row["name"], lab)

    oth = pd.read_excel(XLS_PATH, sheet_name="Other Species", engine="xlrd", header=None, skiprows=1)
    oth.columns = ["name", "cas", "salmonella", "td50", "site"]
    oth = oth.dropna(subset=["cas"])
    for _, row in oth.iterrows():
        c = classify_td50(row["td50"])
        add_call(row["cas"], row["name"], 1 if c == "positive" else (0 if c == "negative" else None))

    return {cas: {"name": name_map[cas], "label": lab} for cas, lab in label_map.items()}


def fetch_smiles(cas: str) -> str | None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(cas)}/property/CanonicalSMILES/TXT"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8").strip() or None
    except Exception:
        return None


def main():
    print("Extracting labels from all four species sheets...")
    compounds = extract_labels()
    print(f"{len(compounds)} labeled compounds (pre-SMILES-resolution).")

    print("Resolving SMILES via PubChem (rate-limited, this takes a while)...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["name", "cas", "carcinogenicity_label", "smiles"])
        found = 0
        for i, (cas, info) in enumerate(compounds.items()):
            smiles = fetch_smiles(cas)
            if smiles:
                writer.writerow([info["name"], cas, info["label"], smiles])
                found += 1
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(compounds)} processed ({found} resolved)")
            time.sleep(PUBCHEM_DELAY)

    print(f"Done. {found}/{len(compounds)} resolved to SMILES. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
