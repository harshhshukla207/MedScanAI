import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from extractor import extract_lab_data

# Initialize FastAPI backend
app = FastAPI(title="Clinical Lab Report Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client — explicitly pass the API key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Check your .env file.")
client = genai.Client(api_key=api_key)

MODELS_DIR = Path(__file__).parent / "saved_models"
PARAMS_CSV = Path(__file__).parent / "parameters.csv"

# --- Correct filename mapping (matches what train_models.py actually saves) ---
DISEASE_MODEL_FILES = {
    "Diabetes":             "diabetes_model.joblib",
    "Anemia":               "anemia_model.joblib",
    "CKD":                  "ckd_model.joblib",
    "Cardiovascular_Risk":  "cardio_model.joblib",
    "Liver_Disease":        "liver_model.joblib",
    "Thyroid":              "thyroid_model.joblib",
}

# --- Human-readable disease names for output messages ---
DISEASE_DISPLAY_NAMES = {
    "Diabetes": "Diabetes",
    "Anemia": "Anemia",
    "CKD": "Chronic Kidney Disease (CKD)",
    "Cardiovascular_Risk": "Cardiovascular Risk",
    "Liver_Disease": "Liver Disease",
    "Thyroid": "Thyroid Disorder",
}

# --- Severity mapping for 3-class output ---
SEVERITY_MAP = {0: "Normal", 1: "Mild", 2: "Chronic"}

# --- Gemini extraction key → training column name mapping ---
EXTRACTION_TO_FEATURE = {
    "hba1c":            "HbA1c",
    "fasting_glucose":  "Fasting_Glucose",
    "egfr":             "eGFR",
    "serum_creatinine": "Serum_Creatinine",
    "ldl":              "LDL",
    "hdl":              "HDL",
    "triglycerides":    "Triglycerides",
    "hemoglobin":       "Hemoglobin",
    "alt_sgpt":         "ALT",
    "ast_sgot":         "AST",
    "tsh":              "TSH",
    "rbc":              "RBC",
    "mcv":              "MCV",
    "mch":              "MCH",
    "ferritin":         "Ferritin",
    "vitamin_b12":      "Vitamin_B12",
    "bun":              "BUN",
    "uric_acid":        "Uric_Acid",
    "hscrp":            "hsCRP",
    "alp":              "ALP",
    "ggt":              "GGT",
    "total_bilirubin":  "Total_Bilirubin",
    "free_t3":          "Free_T3",
    "free_t4":          "Free_T4",
    "age":              "Age",
    "gender":           "Gender",
}

# --- parameters.csv column_name → Gemini extraction key (for mismatched names) ---
# These handle cases where parameters.csv uses a different column_name than
# what EXTRACTION_TO_FEATURE expects (e.g., "rbc_count" vs "rbc").
PARAMS_CSV_TO_GEMINI_KEY = {
    "rbc_count": "rbc",
    "serum_ferritin": "ferritin",
    "alk_phosphatase": "alp",
}

# --- parameters.csv disease name → DISEASE_MODEL_FILES key ---
PARAMS_DISEASE_TO_MODEL_KEY = {
    "Diabetes": "Diabetes",
    "Anemia": "Anemia",
    "CKD": "CKD",
    "Cardiovascular_Risk": "Cardiovascular_Risk",
    "Liver_Disease": "Liver_Disease",
    "Thyroid_Disorder": "Thyroid",
}


# ---------------------------------------------------------------------------
# Smart Model Filtering: Disease → Required Parameters
# ---------------------------------------------------------------------------
def build_disease_required_params(params_csv_path: Path) -> dict[str, list[str]]:
    """
    Read parameters.csv and build a mapping of each disease to its strictly
    required Gemini extraction keys (filtered to only those the ML model uses).

    Returns: {model_disease_key: [list of gemini extraction keys]}
    """
    df = pd.read_csv(params_csv_path)
    disease_params = df.groupby("disease")["column_name"].apply(list).to_dict()

    required = {}
    for csv_disease, columns in disease_params.items():
        model_key = PARAMS_DISEASE_TO_MODEL_KEY.get(csv_disease)
        if model_key is None:
            continue

        gemini_keys = []
        for col in columns:
            # Map to gemini key (handle naming mismatches)
            gemini_key = PARAMS_CSV_TO_GEMINI_KEY.get(col, col)
            # Only include if this key exists in our extraction mapping
            if gemini_key in EXTRACTION_TO_FEATURE:
                gemini_keys.append(gemini_key)

        required[model_key] = gemini_keys

    return required


# --- Primary diagnostic parameters for each disease ---
# The ML models can handle NaNs for secondary parameters, so we only strictly
# require these core parameters to allow a prediction to run.
PRIMARY_DISEASE_PARAMS = {
    "Diabetes": ["hba1c"],
    "Anemia": ["hemoglobin"],
    "CKD": ["egfr"],
    "Cardiovascular_Risk": ["ldl"],
    "Liver_Disease": ["alt_sgpt"],
    "Thyroid": ["tsh"],
}


def validate_disease_params(
    extracted_data: dict,
    required_params: dict[str, list[str]],
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """
    Check which diseases have their PRIMARY parameters present in extracted data.

    Returns:
        can_predict: {disease: True/False}
        missing: {disease: [list of missing training column names for display]}
    """
    extracted_keys = {k for k, v in extracted_data.items() if v is not None}

    can_predict = {}
    missing = {}

    for disease, required_keys in required_params.items():
        primary_keys = PRIMARY_DISEASE_PARAMS.get(disease, required_keys)
        
        # We can predict if AT LEAST ONE primary key is present
        has_primary = any(pk in extracted_keys for pk in primary_keys)
        
        if has_primary:
            can_predict[disease] = True
        else:
            can_predict[disease] = False
            missing[disease] = [EXTRACTION_TO_FEATURE.get(k, k) for k in primary_keys]

    return can_predict, missing


# ---------------------------------------------------------------------------
# Model & Feature Loading
# ---------------------------------------------------------------------------

# Load trained XGBoost models dynamically (extract model from saved dict)
trained_models = {}
feature_columns = []  # Ordered feature list from training

# Load feature column order
feature_cols_path = MODELS_DIR / "feature_columns.json"
if feature_cols_path.exists():
    with open(feature_cols_path) as f:
        feature_columns = json.load(f)

for disease, filename in DISEASE_MODEL_FILES.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        artifact = joblib.load(model_path)
        # train_models.py saves {"model": XGBClassifier, "feature_cols": [...], ...}
        trained_models[disease] = artifact["model"]
        if not feature_columns and "feature_cols" in artifact:
            feature_columns = artifact["feature_cols"]

# Build disease → required parameters mapping from parameters.csv
disease_required_params = {}
if PARAMS_CSV.exists():
    disease_required_params = build_disease_required_params(PARAMS_CSV)


# Pydantic Schema for Strict Gemini Extraction
class LabReportData(BaseModel):
    hba1c: Optional[float] = None
    fasting_glucose: Optional[float] = None
    egfr: Optional[float] = None
    serum_creatinine: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    hemoglobin: Optional[float] = None
    alt_sgpt: Optional[float] = None
    ast_sgot: Optional[float] = None
    tsh: Optional[float] = None
    rbc: Optional[float] = None
    mcv: Optional[float] = None
    mch: Optional[float] = None
    ferritin: Optional[float] = None
    vitamin_b12: Optional[float] = None
    bun: Optional[float] = None
    uric_acid: Optional[float] = None
    hscrp: Optional[float] = None
    alp: Optional[float] = None
    ggt: Optional[float] = None
    total_bilirubin: Optional[float] = None
    free_t3: Optional[float] = None
    free_t4: Optional[float] = None
    age: Optional[float] = None
    gender: Optional[float] = None  # 1=M, 0=F


def align_features_for_prediction(extracted: dict) -> pd.DataFrame:
    """
    Map Gemini-extracted keys to training column names and build a DataFrame
    with ALL 26 feature columns in the correct order. Missing values are NaN
    (XGBoost handles NaN natively).
    """
    mapped = {}
    for gemini_key, value in extracted.items():
        if gemini_key in EXTRACTION_TO_FEATURE and value is not None:
            training_col = EXTRACTION_TO_FEATURE[gemini_key]
            mapped[training_col] = value

    # Build row with all feature columns; missing = NaN
    row = {col: mapped.get(col, np.nan) for col in feature_columns}
    return pd.DataFrame([row])


@app.get("/")
def health_check():
    return {
        "status": "online",
        "models_loaded": list(trained_models.keys()),
        "feature_columns_count": len(feature_columns),
        "disease_required_params": {
            k: v for k, v in disease_required_params.items()
        },
    }


@app.get("/api/parameters")
def get_parameters():
    """Serve parameter reference ranges from parameters.csv."""
    if not PARAMS_CSV.exists():
        raise HTTPException(status_code=404, detail="parameters.csv not found")
    df = pd.read_csv(PARAMS_CSV)
    return df.to_dict(orient="records")


@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # 1. Local OCR Extraction (replaces Gemini Vision API call)
        #    Uses RapidOCR + regex parsing — no network round-trip required.
        extracted_raw = extract_lab_data(image_bytes)
        valid_data = {k: v for k, v in extracted_raw.items() if v is not None}

        if not valid_data:
            return {
                "status": "warning",
                "message": "No valid laboratory parameters could be identified from the uploaded image.",
                "extracted_data": {},
                "predictions": {},
                "unassessed": {}
            }

        # 2. Validate required parameters for each disease
        can_predict, missing_params = validate_disease_params(
            valid_data, disease_required_params
        )

        # 3. Selective prediction: only run models with complete required data
        df_input = align_features_for_prediction(valid_data)
        ml_predictions = {}
        unassessed = {}

        for disease, model in trained_models.items():
            display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)

            if can_predict.get(disease, False):
                # All required parameters present — run prediction
                try:
                    pred = int(model.predict(df_input)[0])
                    ml_predictions[disease] = SEVERITY_MAP.get(pred, f"Unknown ({pred})")
                except Exception:
                    ml_predictions[disease] = "Prediction Error"
            else:
                # Missing required parameters — skip prediction entirely
                missing_list = missing_params.get(disease, [])
                missing_str = ", ".join(missing_list)
                unassessed[disease] = (
                    f"For the analysis of {display_name}, "
                    f"the values of {missing_str} are also required."
                )

        # 4. Clinical Summary Generation via Gemini
        # Map extracted keys to readable names for the summary
        readable_data = {}
        for k, v in valid_data.items():
            if k in EXTRACTION_TO_FEATURE:
                readable_data[EXTRACTION_TO_FEATURE[k]] = v
            else:
                readable_data[k] = v

        summary_prompt = f"""
        SYSTEM INSTRUCTIONS — YOU MUST FOLLOW ALL OF THESE RULES WITHOUT EXCEPTION:

        RULE 1 — FOCUS ON ABNORMALITIES FIRST:
        Begin the summary by prominently highlighting any lab values that fall outside
        normal clinical ranges and any ML model predictions that returned "Mild" or
        "Chronic" risk levels. These abnormal findings must be the most visible part
        of the summary. Normal results should be acknowledged briefly afterwards.

        RULE 2 — SIMPLIFY ALL MEDICAL TERMINOLOGY:
        Every medical term, lab parameter name, and clinical abbreviation MUST be
        explained in simple, plain-English language that a non-medical person can
        understand. For example: "HbA1c (a measure of your average blood sugar over
        the past 3 months)" or "eGFR (a score that shows how well your kidneys are
        filtering waste)". Do NOT assume the reader knows any medical jargon.

        RULE 3 — STRICT NON-DIAGNOSTIC CONSTRAINT:
        Do NOT make a medical diagnosis. You must explicitly state that these results
        are predictive assessments generated by a machine learning model and are NOT
        definitive diagnoses. Use language like "the model's assessment suggests" or
        "based on the predictive analysis" — never "you have" or "this confirms."

        RULE 4 — MANDATORY CLINICAL RECOMMENDATION:
        If ANY abnormal lab values or ANY "Mild" or "Chronic" risk predictions are
        present, you MUST explicitly and strongly recommend that the patient discuss
        these significant findings with a qualified clinician or specialist. Name the
        exact specialist type relevant to the findings (e.g., Endocrinologist for
        diabetes-related findings, Nephrologist for kidney concerns, Hepatologist for
        liver issues, Cardiologist for cardiovascular risk, Hematologist for anemia,
        Endocrinologist for thyroid disorders).

        RULE 5 — MENTION UNASSESSED DISEASES:
        Clearly list any diseases that could not be evaluated due to missing lab
        values, and specify exactly which additional tests would be needed.

        RULE 6 — MEDICAL DISCLAIMER:
        End with a prominent disclaimer: "⚕️ Important: MedScanAI is an informational
        assistant powered by AI and machine learning. It does NOT provide medical
        diagnoses. Always consult a qualified healthcare professional for medical
        advice, diagnosis, or treatment decisions."

        ---
        DATA FOR THIS REPORT:

        Extracted Lab Parameters: {readable_data}
        ML Model Risk Predictions: {ml_predictions}
        Unassessed Diseases (missing data): {unassessed}
        """

        summary_res = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=summary_prompt
        )

        return {
            "status": "success",
            "extracted_data": valid_data,
            "predictions": ml_predictions,
            "unassessed": unassessed,
            "clinical_summary": summary_res.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# NEW: Extract-Only Endpoint (Phase 1 of the review flow)
# ---------------------------------------------------------------------------
@app.post("/extract-report")
async def extract_report(file: UploadFile = File(...)):
    """
    Phase 1: Upload image → OCR extraction only.
    Returns extracted key-value pairs so the user can review/edit before analysis.
    """
    try:
        image_bytes = await file.read()
        extracted_raw = extract_lab_data(image_bytes)
        valid_data = {k: v for k, v in extracted_raw.items() if v is not None}

        if not valid_data:
            return {
                "status": "warning",
                "message": "No valid laboratory parameters could be identified from the uploaded image.",
                "extracted_data": {},
            }

        return {
            "status": "success",
            "extracted_data": valid_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# ---------------------------------------------------------------------------
# NEW: Analyze from User-Edited Data (Phase 2 of the review flow)
# ---------------------------------------------------------------------------
@app.post("/analyze-data")
async def analyze_data(request: Request):
    """
    Phase 2: Accept user-reviewed/edited extracted data as JSON,
    run ML predictions and Gemini clinical summary.
    """
    try:
        body = await request.json()
        valid_data = {k: v for k, v in body.items() if v is not None}

        if not valid_data:
            return {
                "status": "warning",
                "message": "No valid parameters provided for analysis.",
                "extracted_data": {},
                "predictions": {},
                "unassessed": {},
            }

        # 1. Validate required parameters for each disease
        can_predict, missing_params = validate_disease_params(
            valid_data, disease_required_params
        )

        # 2. Selective prediction: only run models with complete required data
        df_input = align_features_for_prediction(valid_data)
        ml_predictions = {}
        unassessed = {}

        for disease, model in trained_models.items():
            display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)

            if can_predict.get(disease, False):
                try:
                    pred = int(model.predict(df_input)[0])
                    ml_predictions[disease] = SEVERITY_MAP.get(pred, f"Unknown ({pred})")
                except Exception:
                    ml_predictions[disease] = "Prediction Error"
            else:
                missing_list = missing_params.get(disease, [])
                missing_str = ", ".join(missing_list)
                unassessed[disease] = (
                    f"For the analysis of {display_name}, "
                    f"the values of {missing_str} are also required."
                )

        # 3. Clinical Summary Generation via Gemini
        readable_data = {}
        for k, v in valid_data.items():
            if k in EXTRACTION_TO_FEATURE:
                readable_data[EXTRACTION_TO_FEATURE[k]] = v
            else:
                readable_data[k] = v

        summary_prompt = f"""
        SYSTEM INSTRUCTIONS — YOU MUST FOLLOW ALL OF THESE RULES WITHOUT EXCEPTION:

        RULE 1 — FOCUS ON ABNORMALITIES FIRST:
        Begin the summary by prominently highlighting any lab values that fall outside
        normal clinical ranges and any ML model predictions that returned "Mild" or
        "Chronic" risk levels. These abnormal findings must be the most visible part
        of the summary. Normal results should be acknowledged briefly afterwards.

        RULE 2 — SIMPLIFY ALL MEDICAL TERMINOLOGY:
        Every medical term, lab parameter name, and clinical abbreviation MUST be
        explained in simple, plain-English language that a non-medical person can
        understand. For example: "HbA1c (a measure of your average blood sugar over
        the past 3 months)" or "eGFR (a score that shows how well your kidneys are
        filtering waste)". Do NOT assume the reader knows any medical jargon.

        RULE 3 — STRICT NON-DIAGNOSTIC CONSTRAINT:
        Do NOT make a medical diagnosis. You must explicitly state that these results
        are predictive assessments generated by a machine learning model and are NOT
        definitive diagnoses. Use language like "the model's assessment suggests" or
        "based on the predictive analysis" — never "you have" or "this confirms."

        RULE 4 — MANDATORY CLINICAL RECOMMENDATION:
        If ANY abnormal lab values or ANY "Mild" or "Chronic" risk predictions are
        present, you MUST explicitly and strongly recommend that the patient discuss
        these significant findings with a qualified clinician or specialist. Name the
        exact specialist type relevant to the findings (e.g., Endocrinologist for
        diabetes-related findings, Nephrologist for kidney concerns, Hepatologist for
        liver issues, Cardiologist for cardiovascular risk, Hematologist for anemia,
        Endocrinologist for thyroid disorders).

        RULE 5 — MENTION UNASSESSED DISEASES:
        Clearly list any diseases that could not be evaluated due to missing lab
        values, and specify exactly which additional tests would be needed.

        RULE 6 — MEDICAL DISCLAIMER:
        End with a prominent disclaimer: "⚕️ Important: MedScanAI is an informational
        assistant powered by AI and machine learning. It does NOT provide medical
        diagnoses. Always consult a qualified healthcare professional for medical
        advice, diagnosis, or treatment decisions."

        ---
        DATA FOR THIS REPORT:

        Extracted Lab Parameters: {readable_data}
        ML Model Risk Predictions: {ml_predictions}
        Unassessed Diseases (missing data): {unassessed}
        """

        summary_res = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=summary_prompt
        )

        return {
            "status": "success",
            "extracted_data": valid_data,
            "predictions": ml_predictions,
            "unassessed": unassessed,
            "clinical_summary": summary_res.text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# --- Serve static frontend ---
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/app")
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not built yet")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")