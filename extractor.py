"""
extractor.py — Local OCR-based Lab Report Data Extraction

Uses RapidOCR (ONNX Runtime) for text extraction and regex-based
parameter parsing to replace the Gemini Vision API for the initial
data extraction phase.  This removes one network round-trip and
drastically reduces latency.

Pipeline:
    image bytes → RapidOCR → raw text → regex parse → dict[str, float]
"""

import re
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# ---------------------------------------------------------------------------
# OCR Engine (module-level singleton — loaded once at import time)
# ---------------------------------------------------------------------------
ocr_engine = RapidOCR()

PARAMS_CSV = Path(__file__).parent / "parameters.csv"

# ---------------------------------------------------------------------------
# Parameter Alias Patterns
# ---------------------------------------------------------------------------
# Each key matches a field in the LabReportData Pydantic schema.
# Aliases are regex patterns matched case-insensitively.
# Within each list, MORE SPECIFIC patterns come first so they are tried
# before shorter, ambiguous alternatives.
#
# NOTE: Age and Gender are handled separately with specialised logic.
# ---------------------------------------------------------------------------
PARAMETER_PATTERNS: dict[str, list[str]] = {
    "hba1c": [
        r"hba1c",
        r"hb\s*a1c",
        r"glycated\s*h[ae]moglobin",
        r"glycosylated\s*h[ae]moglobin",
        r"(?<![a-z])a1c(?![a-z])",
    ],
    "fasting_glucose": [
        r"fasting\s*(?:plasma\s*)?glucose",
        r"fasting\s*blood\s*sugar",
        r"(?<![a-z])fbs(?![a-z])",
        r"(?<![a-z])fpg(?![a-z])",
        r"fasting\s*sugar",
    ],
    "egfr": [
        r"estimated\s*(?:glomerular\s*)?filtration\s*rate",
        r"e[\-\s]?gfr",
    ],
    "serum_creatinine": [
        r"serum\s*creatinine",
        r"s[\.\s]*creatinine",
        r"creatinine",
    ],
    "ldl": [
        r"ldl[\-\s]*cholesterol",
        r"low\s*density\s*lipoprotein",
        r"ldl[\-\s]*c(?![a-z])",
        r"(?<![a-z])ldl(?![a-z])",
    ],
    "hdl": [
        r"hdl[\-\s]*cholesterol",
        r"high\s*density\s*lipoprotein",
        r"hdl[\-\s]*c(?![a-z])",
        r"(?<![a-z])hdl(?![a-z])",
    ],
    "triglycerides": [
        r"triglycerides?",
        r"serum\s*triglycerides?",
        r"(?<![a-z])tg(?![a-z])",
    ],
    "hemoglobin": [
        # Must NOT match "HbA1c" — negative lookahead for a1c
        r"h[ae]moglobin(?!\s*a1c)",
        r"(?<![a-z])hgb(?![a-z])",
        r"(?<![a-z])hb(?!\s*a1c)(?![a-z])",
    ],
    "alt_sgpt": [
        r"alt\s*[\(\[/]?\s*sgpt",
        r"sgpt\s*[\(\[/]?\s*alt",
        r"alanine\s*(?:amino\s*)?transferase",
        r"(?<![a-z])sgpt(?![a-z])",
        r"(?<![a-z])alt(?![a-z])",
    ],
    "ast_sgot": [
        r"ast\s*[\(\[/]?\s*sgot",
        r"sgot\s*[\(\[/]?\s*ast",
        r"aspartate\s*(?:amino\s*)?transferase",
        r"(?<![a-z])sgot(?![a-z])",
        r"(?<![a-z])ast(?![a-z])",
    ],
    "tsh": [
        r"thyroid\s*stimulating\s*hormone",
        r"thyrotropin",
        r"(?<![a-z])tsh(?![a-z])",
    ],
    "rbc": [
        r"red\s*blood\s*cell(?:\s*count)?",
        r"erythrocyte\s*count",
        r"total\s*rbc",
        r"(?<![a-z])rbc\s*count",
        r"(?<![a-z])rbc(?![a-z])",
    ],
    "mcv": [
        r"mean\s*corpuscular\s*volume",
        r"(?<![a-z])mcv(?![a-z])",
    ],
    "mch": [
        # Must NOT match "MCHC"
        r"mean\s*corpuscular\s*h[ae]moglobin(?!\s*con)",
        r"(?<![a-z])mch(?!c)(?![a-z])",
    ],
    "ferritin": [
        r"serum\s*ferritin",
        r"s[\.\s]*ferritin",
        r"ferritin",
    ],
    "vitamin_b12": [
        r"vitamin\s*b[\-\s]?12",
        r"vit[\.\s]*b[\-\s]?12",
        r"cyanocobalamin",
        r"cobalamin",
        r"(?<![a-z])b[\-\s]?12(?![0-9])",
    ],
    "bun": [
        r"blood\s*urea\s*nitrogen",
        r"urea\s*nitrogen",
        r"(?<![a-z])bun(?![a-z])",
    ],
    "uric_acid": [
        r"serum\s*uric\s*acid",
        r"uric\s*acid",
        r"urate",
    ],
    "hscrp": [
        r"hs[\-\s]?crp",
        r"high[\-\s]*sensitivity\s*c[\-\s]?reactive\s*protein",
        r"c[\-\s]?reactive\s*protein",
        r"(?<![a-z])crp(?![a-z])",
    ],
    "alp": [
        r"alkaline\s*phosphatase",
        r"alk[\.\s]*phos(?:phatase)?",
        r"(?<![a-z])alp(?![a-z])",
    ],
    "ggt": [
        r"gamma[\-\s]*glutamyl\s*transferase",
        r"gamma[\-\s]*gt",
        r"(?<![a-z])ggt(?![a-z])",
    ],
    "total_bilirubin": [
        r"total\s*bilirubin",
        r"t[\.\s]+bilirubin",
        r"bilirubin\s*[\(\[]?\s*total\s*[\)\]]?",
        r"serum\s*bilirubin",
        r"bilirubin",  # fallback — standalone usually means total
    ],
    "free_t3": [
        r"free\s*t[\-\s]?3",
        r"free\s*triiodothyronine",
        r"(?<![a-z])ft3(?![a-z])",
    ],
    "free_t4": [
        r"free\s*t[\-\s]?4",
        r"free\s*thyroxine",
        r"(?<![a-z])ft4(?![a-z])",
    ],
}


# ---------------------------------------------------------------------------
# Augment aliases with display_name values from parameters.csv
# ---------------------------------------------------------------------------
def _load_csv_display_aliases(csv_path: Path) -> dict[str, list[str]]:
    """
    Extract the primary display name from parameters.csv for each
    column_name that maps to one of our target extraction keys.
    Returns {extraction_key: [regex_escaped_display_name, ...]}.
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    # Bridge from parameters.csv column_name to extraction key
    params_bridge = {"rbc_count": "rbc", "serum_ferritin": "ferritin", "alk_phosphatase": "alp"}
    target_keys = set(PARAMETER_PATTERNS.keys())

    csv_aliases: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        col = str(row.get("column_name", ""))
        display = str(row.get("display_name", ""))
        key = params_bridge.get(col, col)
        if key not in target_keys or not display:
            continue

        # Extract primary name (text before the first parenthesis / dash / slash)
        primary = re.split(r"[\(\[—/]", display)[0].strip()
        if len(primary) > 2 and (key, primary) not in seen:
            seen.add((key, primary))
            csv_aliases.setdefault(key, []).append(re.escape(primary))

    return csv_aliases


_csv_aliases = _load_csv_display_aliases(PARAMS_CSV)
for _key, _aliases in _csv_aliases.items():
    if _key in PARAMETER_PATTERNS:
        # Append CSV-derived patterns AFTER the manually-curated ones
        PARAMETER_PATTERNS[_key].extend(_aliases)


# ---------------------------------------------------------------------------
# Pre-compile all regex patterns (done once at import time)
# ---------------------------------------------------------------------------
COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    param: [re.compile(p, re.IGNORECASE) for p in patterns]
    for param, patterns in PARAMETER_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def ocr_image(image_bytes: bytes) -> str:
    """Run RapidOCR on raw image bytes and return all recognised text lines."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    result, _ = ocr_engine(img_array)

    if not result:
        return ""

    # Sort OCR results by vertical position (top-to-bottom, left-to-right)
    try:
        result.sort(key=lambda r: (min(p[1] for p in r[0]),
                                   min(p[0] for p in r[0])))
    except (IndexError, TypeError):
        pass  # If bounding-box sort fails, use original order

    lines = []
    for item in result:
        if len(item) >= 2 and isinstance(item[1], str):
            lines.append(item[1])

    return "\n".join(lines)


def _extract_number_after(text: str, start: int, window: int = 120) -> float | None:
    """
    Search for the first plausible numerical value within *window* characters
    after position *start* in *text*.

    Stops at the next newline to avoid crossing into another parameter's row.

    Handles formats like:
        ": 5.7 %"  |  "= 12.5"  |  "  95 mg/dL"  |  "5.7%"
    """
    segment = text[start : start + window]

    # Don't cross into the next line
    nl = segment.find("\n")
    if nl > 0:
        segment = segment[:nl]

    m = re.search(r"(\d+\.?\d*)", segment)
    if m:
        try:
            val = float(m.group(1))
            # Reject values that look like years, IDs, or phone numbers
            if val > 10_000:
                return None
            return val
        except ValueError:
            return None
    return None


def parse_parameters(raw_text: str) -> dict[str, float]:
    """
    Scan OCR text for the 26 target lab parameters using compiled regex
    aliases.  Returns a dict keyed by LabReportData field names with float
    values for every parameter that was successfully matched and extracted.
    """
    text_lower = raw_text.lower()
    extracted: dict[str, float] = {}

    for param_key, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text_lower)
            if match:
                value = _extract_number_after(text_lower, match.end())
                if value is not None:
                    extracted[param_key] = value
                    break  # Found this parameter — move to the next

    # --- Special handling: Age ---
    age_match = re.search(r"\bage\s*[:/\-]?\s*(\d{1,3})\b", text_lower)
    if not age_match:
        age_match = re.search(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*(?:old)?", text_lower)
    if age_match:
        age_val = int(age_match.group(1))
        if 0 < age_val <= 120:
            extracted["age"] = float(age_val)

    # --- Special handling: Gender ---
    gender_match = re.search(
        r"\b(?:sex|gender)\s*[:/\-]?\s*(male|female|m|f)\b", text_lower
    )
    if not gender_match:
        gender_match = re.search(r"\b(male|female)\b", text_lower)
    if gender_match:
        g = gender_match.group(1)
        extracted["gender"] = 1.0 if g in ("male", "m") else 0.0

    return extracted


def extract_lab_data(image_bytes: bytes) -> dict[str, float]:
    """
    Full local extraction pipeline:
      1. OCR the image using RapidOCR (ONNX Runtime)
      2. Parse the raw text for 26 lab parameters via regex
      3. Return a dict matching the LabReportData schema fields
    """
    raw_text = ocr_image(image_bytes)
    return parse_parameters(raw_text)
