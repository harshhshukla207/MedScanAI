# MedScanAI

MedScanAI is an AI-powered Clinical Lab Report Analyzer that leverages Optical Character Recognition (OCR), Machine Learning (XGBoost), and Large Language Models (LLMs) to automatically extract, analyze, and interpret medical laboratory reports. 

The system reads an uploaded image of a lab report, extracts critical clinical parameters locally without network overhead, predicts the severity of 6 major diseases, and generates a plain-English, non-diagnostic clinical summary using the Gemini API.

> **⚕️ Important Disclaimer:** MedScanAI is an informational assistant powered by AI and machine learning. It does NOT provide medical diagnoses. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment decisions.

## Features

- **Local OCR Extraction:** Uses `RapidOCR` (ONNX Runtime) and regex-based parsing to extract 26 key clinical parameters from report images. This local-first approach ensures privacy and reduces latency compared to cloud-vision APIs.
- **Machine Learning Disease Prediction:** Uses trained XGBoost classifiers to assess severity (Normal, Mild, Chronic) for 6 major conditions:
  - Diabetes
  - Anemia
  - Chronic Kidney Disease (CKD)
  - Cardiovascular Risk
  - Liver Disease
  - Thyroid Disorder
- **LLM Clinical Summary:** Integrates with Google's Gemini API (`gemini-3.6-flash`) to generate easy-to-understand, plain-English summaries that highlight abnormal values and recommend appropriate specialist consultations.
- **Two-Phase Review Flow:** 
  1. Extract raw data from the image.
  2. Allow the user to review/edit the extracted data before submitting it for ML analysis and summary generation.

## Architecture

The project consists of three main components:

1. **`app.py`**: A FastAPI backend that handles file uploads, orchestrates the OCR extraction, validates required parameters for ML models, runs the XGBoost predictions, and calls the Gemini API to construct the final clinical summary.
2. **`extractor.py`**: The local OCR engine powered by RapidOCR. It parses the raw text using highly-specific regex patterns tailored for medical parameters, handling aliases and units gracefully.
3. **`train_models.py`**: The machine learning training pipeline. It processes a dataset (`comprehensive_clinical_data.csv`), handles target reclassification (Binary to 3-class severity based on clinical thresholds), balances classes, and trains 6 XGBoost classifiers.

### Prerequisites
- Python 3.8+
- [Gemini API Key] for clinical summary generation.
