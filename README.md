# AI Diabetes Laboratory Report Assistant

## Project overview
This educational Streamlit application accepts diabetes-related laboratory reports as PDFs or images. It extracts HbA1c and blood-glucose results, checks extracted values, and uses a LangGraph workflow to decide whether to complete processing, retry extraction, or request manual review. When validation permits and the Groq API is configured, it can generate an educational explanation.

**Safety notice:** This application does not diagnose diabetes and must not be used to make treatment decisions. Always compare extracted values with the original report and consult a qualified healthcare professional.

## Main capabilities
- PDF and image report upload.
- PDF text extraction and image OCR.
- Image preprocessing and retry when extraction is uncertain.
- Validation checks that flag suspicious values.
- LangGraph routing to completion, retry, or manual review.
- No silent correction of suspicious extracted values.
- Optional Groq-powered educational explanation.
- Workflow status and retry count shown in Streamlit.
- Report date and laboratory reference range are captured only when explicitly present and identifiable in the report; otherwise the app displays “Not identified.”

## Requirements
- Python compatible with `requirements.txt`.
- Tesseract OCR installed and configured for `pytesseract`.
- Dependencies listed in `requirements.txt`.
- Groq API key for the optional LLM-generated explanation.

## Setup on Windows
1. Open PowerShell in the project folder.
2. Create and activate a virtual environment if needed:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your own key using the variable name expected by the application. The current setup is expected to use:

   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

   Never commit or share the real key. If the code expects a different variable name, follow the name used in the code.

5. Ensure Tesseract is installed and correctly configured. If the application uses a Windows executable path or `TESSDATA_PREFIX`, update it for your machine.

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Run integration tests
From the project root, with the virtual environment active:

```powershell
python -m unittest tests.test_langgraph_integration -v
```

The integration and metadata tests cover normal completion, retry disagreement, suspicious-value manual review, unreadable-report retry/manual review, explicit date/reference-range extraction, and missing-metadata handling. They mock the extraction function and verify workflow routing/state handling; they do not measure OCR accuracy. The three synthetic sample reports were also checked separately through the Streamlit UI. The automated workflow tests mock file extraction and do not measure OCR accuracy; sample-by-sample OCR accuracy should be recorded separately against manually verified expected values.

## Suggested project structure
Keep the actual folder layout as it currently exists. A typical layout includes:

```text
DiabetesReportAssistant_Final/
├── app.py
├── requirements.txt
├── .env                 # local secret; do not submit/share
├── .gitignore
├── extraction/
├── validation/
├── workflow/
└── tests/
```

## Limitations
- OCR can misread decimal points, units, labels, or numbers.
- Plausibility checks are not clinical interpretation.
- Reference ranges vary by laboratory and patient context.
- LLM-generated explanations can be incomplete or incorrect.
- Uncertain extraction should result in manual review, not invented values.
