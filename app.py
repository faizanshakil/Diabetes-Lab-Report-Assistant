import html
import pandas as pd
import streamlit as st

from workflow.explainer import explain_results
from workflow.graph import report_graph

st.set_page_config(
    page_title="GlucoGuide | Diabetes Report Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-contrast navy, ivory, and amber theme. The result navigation uses a
# horizontal radio selector styled as segmented buttons instead of Streamlit
# tabs, so all section labels remain readable across Streamlit versions.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --page: #f3f1eb;
        --surface: #ffffff;
        --ink: #172033;
        --muted: #4c5b70;
        --navy: #14243a;
        --navy2: #203b5b;
        --teal: #087e83;
        --teal-dark: #075d62;
        --amber: #f4b544;
        --line: #cbd3dd;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(243,241,235,.96); }
    .block-container { max-width: 1240px; padding-top: 1.5rem; padding-bottom: 3rem; }

    [data-testid="stSidebar"] { background: var(--navy); border-right: 3px solid var(--amber); }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] hr { border-color: #53647a; }
    .brand { font: 800 1.18rem Manrope, sans-serif; letter-spacing: -.04em; color: #fff; padding: .5rem 0 1.3rem; }
    .brand span { color: var(--amber); }
    .side-label { color: #f4c66f !important; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 800; margin-top: 1rem; }
    .side-copy { color: #e2e8f0 !important; font-size: .9rem; line-height: 1.7; }

    .hero {
        background: linear-gradient(120deg, #14243a 0%, #203b5b 62%, #087e83 100%);
        border: 1px solid #284761; border-bottom: 5px solid var(--amber);
        border-radius: 20px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.35rem;
        box-shadow: 0 14px 30px rgba(20,36,58,.15); position: relative; overflow: hidden;
    }
    .hero:after { content: '+'; position: absolute; right: 2rem; top: -2.6rem; font: 800 12rem Manrope,sans-serif; color: rgba(255,255,255,.07); }
    .eyebrow { color: #f4c66f; font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .65rem; }
    .hero h1 { font: 800 clamp(1.8rem,3.4vw,2.65rem) Manrope,sans-serif; line-height: 1.13; letter-spacing: -.045em; margin: 0 0 .8rem; color: #fff !important; }
    .hero p { color: #f1f5f9 !important; font-size: 1rem; line-height: 1.65; max-width: 720px; margin: 0; }

    .section-title { font: 800 1.15rem Manrope,sans-serif; color: var(--ink) !important; letter-spacing: -.025em; margin: .25rem 0 .3rem; }
    .section-sub { color: var(--muted) !important; font-size: .9rem; margin-bottom: 1rem; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-top: 4px solid var(--teal); border-radius: 15px; padding: 1.2rem 1.3rem; box-shadow: 0 5px 16px rgba(20,36,58,.06); margin-bottom: 1rem; }
    .step-chip { display: inline-block; background: #fff0cb; color: #5a3900 !important; padding: .34rem .65rem; border-radius: 6px; font-size: .72rem; font-weight: 800; letter-spacing: .04em; margin-bottom: .65rem; }
    .small-note { font-size: .82rem; line-height: 1.55; color: var(--muted) !important; }

    /* Upload area follows the same navy + amber palette as the sidebar */
    div[data-testid="stFileUploader"] {
        background: #14243a !important;
        border: 2px dashed #f4b544 !important;
        border-radius: 12px !important;
        padding: .7rem !important;
    }
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: #14243a !important;
        border: 0 !important;
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #f8fafc !important;
    }
    div[data-testid="stFileUploader"] button {
        background: #f4b544 !important;
        color: #14243a !important;
        border: 1px solid #ffd477 !important;
        border-radius: 7px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: #ffd477 !important;
        color: #14243a !important;
    }
    /* Uploaded-file chip: blue-navy surface, white text, amber accents */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: #203b5b !important;
        border: 1px solid #f4b544 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] { background: var(--teal) !important; color: #fff !important; border: 2px solid var(--teal-dark) !important; border-radius: 9px; min-height: 3rem; font-weight: 800; box-shadow: 0 4px 10px rgba(8,126,131,.18); }
    .stButton > button[kind="primary"]:hover { background: var(--teal-dark) !important; border-color: var(--teal-dark) !important; }
    .stDownloadButton > button { background: #fff !important; color: var(--navy) !important; border: 1px solid #9aa9b8 !important; border-radius: 8px; font-weight: 700; }

    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-top: 4px solid var(--teal); border-radius: 12px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--navy) !important; font-family: Manrope,sans-serif; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }

    /* Light, readable result surfaces across all three report sections */
    [data-testid="stTextArea"] textarea,
    textarea[data-testid="stTextArea"] {
        background: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #aab8c7 !important;
        border-radius: 9px !important;
        caret-color: #087e83 !important;
    }
    [data-testid="stTextArea"] label,
    [data-testid="stTextArea"] p { color: #172033 !important; }
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        color: #14243a !important;
        border: 1px solid #8798aa !important;
    }
    [data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        background: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #cbd3dd !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stMarkdownContainer"] thead tr th {
        background: #e7f1f3 !important;
        color: #14243a !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #a9c5cb !important;
    }
    [data-testid="stMarkdownContainer"] tbody tr td {
        background: #ffffff !important;
        color: #172033 !important;
        border-bottom: 1px solid #dce3ea !important;
    }
    [data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td {
        background: #f4f7f9 !important;
    }
    .lab-results-table-wrap {
        width: 100%; overflow-x: auto; border: 1px solid #cbd3dd;
        border-radius: 10px; background: #ffffff; margin: .5rem 0;
    }
    table.lab-results-table {
        width: 100%; border-collapse: collapse; background: #ffffff;
        color: #172033; font-size: .92rem;
    }
    .lab-results-table th {
        background: #e7f1f3; color: #14243a; text-align: left;
        padding: .8rem .9rem; border-bottom: 2px solid #a9c5cb;
        font-weight: 800; white-space: nowrap;
    }
    .lab-results-table td {
        background: #ffffff; color: #172033; padding: .75rem .9rem;
        border-bottom: 1px solid #dce3ea; vertical-align: top;
    }
    .lab-results-table tbody tr:nth-child(even) td { background: #f4f7f9; }
    .lab-results-table tbody tr:last-child td { border-bottom: 0; }

    /* Three report-section buttons use the sidebar's navy, white, and amber palette */
    div[role="radiogroup"] { gap: .55rem !important; }
    div[role="radiogroup"] label {
        background: #203b5b !important;
        border: 2px solid #53647a !important;
        border-radius: 9px !important;
        padding: .58rem 1rem !important;
        margin: 0 !important;
        min-height: 44px;
        transition: background .15s ease, border-color .15s ease;
    }
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label div { color: #f8fafc !important; font-weight: 800 !important; }
    div[role="radiogroup"] label:has(input:checked) {
        background: #f4b544 !important;
        border-color: #14243a !important;
        box-shadow: inset 0 -3px 0 #14243a;
    }
    div[role="radiogroup"] label:has(input:checked) p,
    div[role="radiogroup"] label:has(input:checked) span,
    div[role="radiogroup"] label:has(input:checked) div { color: #14243a !important; }
    div[role="radiogroup"] label:hover { background: #2c4c70 !important; border-color: #f4b544 !important; }
    div[role="radiogroup"] label:has(input:checked):hover { background: #ffd477 !important; }

    .footer { border-top: 2px solid #cbd3dd; margin-top: 2rem; padding-top: 1rem; color: #4c5b70 !important; font-size: .8rem; line-height: 1.65; }
    [data-testid="stAlert"] { border-radius: 10px; }
    .warning-notice {
        background: #fff0cb !important;
        color: #4b3200 !important;
        border: 1px solid #d69e2e !important;
        border-left: 5px solid #b7791f !important;
        border-radius: 10px !important;
        padding: .85rem 1rem !important;
        margin: .5rem 0 1rem !important;
        font-weight: 600 !important;
        line-height: 1.55 !important;
    }
    .warning-notice * { color: #4b3200 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_warning(message: str) -> None:
    """Display an accessible amber warning with dark, readable text."""
    st.markdown(
        f'<div class="warning-notice">{html.escape(str(message))}</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand">Gluco<span>Guide</span> <span style="color:#f8fafc;font-weight:500;font-size:.75rem;letter-spacing:0">/ LABS</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">Diabetes laboratory report assistant</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="side-label">What it does</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">• Reads PDF and image reports<br>• Extracts glucose and HbA1c values<br>• Checks results for suspicious formatting<br>• Routes uncertain reports for manual review</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="side-label">Privacy reminder</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">Use de-identified reports. Uploaded reports may contain sensitive health information.</div>', unsafe_allow_html=True)

st.markdown(
    '''<section class="hero">
      <div class="eyebrow">Intelligent report reading · Educational support</div>
      <h1>Understand your lab report.<br>Keep every result verifiable.</h1>
    </section>''',
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, .88], gap="large")
with left:
    st.markdown('<div class="panel"><div class="step-chip">STEP 01 · UPLOAD</div><div class="section-title">Choose a laboratory report</div><div class="section-sub">Supported formats: PDF, PNG, JPG, JPEG</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload report file",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Choose a PDF or a report image. For best results, use a clear, straight image.",
    )
    if uploaded_file is not None:
        st.success(f"Ready to process: {uploaded_file.name} · {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        if uploaded_file.type.startswith("image/"):
            with st.expander("Preview uploaded image", expanded=False):
                st.image(uploaded_file, use_container_width=True)
    process_clicked = st.button("Analyze report  →", type="primary", use_container_width=True, disabled=uploaded_file is None)
    st.markdown('<div class="small-note" style="margin-top:.8rem">Processing may take a little time for scanned documents. Keep the original report available for comparison.</div></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel"><div class="step-chip">STEP 02 · REVIEW</div><div class="section-title">Designed for cautious extraction</div><div class="section-sub">The workflow can retry and request human review.</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="display:flex;gap:.8rem;margin:.9rem 0 1.05rem;align-items:flex-start">
      <div style="background:#d7f1ee;color:#064e52;border-radius:9px;padding:.55rem .7rem;font-size:1.15rem;font-weight:800">1</div>
      <div><b style="color:#172033">Extract</b><div class="small-note">Read embedded PDF text or use OCR for scans and images.</div></div>
    </div>
    <div style="display:flex;gap:.8rem;margin:.9rem 0 1.05rem;align-items:flex-start">
      <div style="background:#e1e9ff;color:#203b5b;border-radius:9px;padding:.55rem .7rem;font-size:1.15rem;font-weight:800">2</div>
      <div><b style="color:#172033">Validate</b><div class="small-note">Check recognized values and formatting; retry when needed.</div></div>
    </div>
    <div style="display:flex;gap:.8rem;margin:.9rem 0 .2rem;align-items:flex-start">
      <div style="background:#fff0cb;color:#5a3900;border-radius:9px;padding:.55rem .7rem;font-size:1.15rem;font-weight:800">!</div>
      <div><b style="color:#172033">Escalate uncertainty</b><div class="small-note">Questionable results are shown for manual review rather than automatically corrected.</div></div>
    </div></div>''', unsafe_allow_html=True)

if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_filename" not in st.session_state:
    st.session_state.report_filename = None
if "report_explanation" not in st.session_state:
    st.session_state.report_explanation = None
if "report_explanation_error" not in st.session_state:
    st.session_state.report_explanation_error = None

if process_clicked and uploaded_file is not None:
    initial_state = {
        "file_bytes": uploaded_file.getvalue(),
        "filename": uploaded_file.name,
        "extracted_text": "",
        "extraction_method": "",
        "lab_results": [],
        "validation_passed": False,
        "retry_count": 0,
        "status": "processing",
        "message": "Starting report processing.",
    }
    try:
        with st.spinner("Reading the report and validating extracted values…"):
            result = report_graph.invoke(initial_state)
        st.session_state.report_result = result
        st.session_state.report_filename = uploaded_file.name
        st.session_state.report_explanation = None
        st.session_state.report_explanation_error = None
    except Exception as error:
        st.error("The workflow encountered an error while processing this report.")
        st.exception(error)

result = st.session_state.report_result
if result:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Report analysis</div><div class="section-sub">Review the workflow outcome and compare every value with the original.</div>', unsafe_allow_html=True)

    status = result.get("status", "unknown")
    validation_passed = result.get("validation_passed") is True
    extracted_text = result.get("extracted_text", "")
    lab_results = result.get("lab_results", [])
    display_status = status.replace("_", " ").title()

    if status == "completed" and validation_passed:
        st.success("Workflow completed · Basic extraction checks passed")
    elif status == "manual_review":
        st.error("Manual review required · Please verify against the original report")
    else:
        show_warning("Workflow did not confirm successful validation")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Workflow status", display_status)
    metric_cols[1].metric("Extraction retries", result.get("retry_count", 0))
    metric_cols[2].metric("Results identified", len(lab_results))
    st.caption(f"**Extraction method:** {result.get('extraction_method', 'Not available')}  ·  **Details:** {result.get('message', 'No message provided.')}")

    selected_section = st.radio(
        "Report sections",
        ["Lab results", "Extracted text", "Educational explanation"],
        horizontal=True,
        label_visibility="collapsed",
        key="report_section_navigation",
    )
    st.markdown("<hr style='border:0;border-top:2px solid #cbd3dd;margin:.35rem 0 1.2rem'>", unsafe_allow_html=True)

    if selected_section == "Lab results":
        if lab_results:
            # Render a custom light table instead of Streamlit's dataframe grid,
            # whose dark theme can make the results look inconsistent.
            columns = list(lab_results[0].keys())
            header_cells = "".join(f"<th>{html.escape(str(column).replace('_', ' ').title())}</th>" for column in columns)
            body_rows = []
            for row in lab_results:
                cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
                body_rows.append(f"<tr>{cells}</tr>")
            table_html = (
                '<div class="lab-results-table-wrap"><table class="lab-results-table">'
                f'<thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody>'
                '</table></div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)
            st.caption("Confirm test names, values, decimal points, and units against the source report.")
        else:
            show_warning("No supported laboratory results were identified. Review the original report manually.")
        if status == "manual_review":
            st.error("Do not rely on these extracted values until they have been checked against the original report. No values were automatically corrected.")
        elif status == "completed" and validation_passed:
            st.info("Basic extraction and formatting checks passed. This does not establish clinical accuracy.")

    elif selected_section == "Extracted text":
        if extracted_text:
            st.text_area("Text recognized from the report", extracted_text, height=260, key="recognized_report_text")
            st.download_button(
                "Download extracted text",
                data=extracted_text,
                file_name=f"{(st.session_state.report_filename or 'report').rsplit('.', 1)[0]}_extracted.txt",
                mime="text/plain",
            )
        else:
            st.info("No extracted text is available for this report.")

    else:
        if status == "completed" and validation_passed and lab_results:
            if st.session_state.report_explanation is None and st.session_state.report_explanation_error is None:
                try:
                    with st.spinner("Preparing an educational explanation…"):
                        st.session_state.report_explanation = explain_results(lab_results)
                except Exception as error:
                    st.session_state.report_explanation_error = str(error)
            if st.session_state.report_explanation:
                st.markdown(st.session_state.report_explanation)
            else:
                show_warning("The values passed basic checks, but the explanation service could not respond. You can still review the extracted values above.")
                with st.expander("Technical details"):
                    st.code(st.session_state.report_explanation_error or "No error details were returned.")
        else:
            show_warning("An educational explanation is withheld because the results did not pass validation. Compare the values and units with the original report first.")

st.markdown(
    '<div class="footer"><b>Important health notice</b><br>This tool supports laboratory report extraction and educational understanding only. It does not diagnose diabetes, recommend treatment, or replace a healthcare professional. OCR may misread decimal points or units. Always verify results against the original laboratory report.</div>',
    unsafe_allow_html=True,
)
