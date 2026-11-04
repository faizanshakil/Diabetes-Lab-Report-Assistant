from copy import deepcopy
import re

from langgraph.graph import StateGraph, START, END

from workflow.state import ReportState
from extraction.reader import extract_report
from validation.result_validator import (
    extract_lab_results,
    validate_lab_results,
)


MAX_RETRIES = 1


# --------------------------------------------------
# INSPECT REPORT
# --------------------------------------------------

def inspect_report(state: ReportState):
    """Initialize processing for a newly uploaded report."""

    return {
        "retry_count": 0,
        "lab_results": [],
        "validation_passed": False,
        "status": "processing",
        "message": "Inspecting uploaded report.",
        "extracted_text": "",
        "extraction_method": "",
    }


# --------------------------------------------------
# EXTRACT TEXT
# --------------------------------------------------

def extract_text(state: ReportState):
    """Extract text from a PDF or image."""

    try:
        result = extract_report(
            state["file_bytes"],
            state["filename"],
        )

        return {
            "extracted_text": result.get("text", ""),
            "extraction_method": result.get(
                "method", "Unknown extraction method"
            ),
            "message": (
                result.get("warning")
                or "Text extraction completed."
            ),
        }

    except Exception as error:
        return {
            "extracted_text": "",
            "extraction_method": "Extraction failed",
            "message": (
                f"Report extraction failed: {error}"
            ),
        }


# --------------------------------------------------
# ALTERNATIVE OCR RETRY
# --------------------------------------------------

def retry_with_ocr(state: ReportState):
    """Retry extraction; resolve unit-only differences conservatively.

    Numeric value/test-name disagreements always trigger manual review and
    retain the original text. If test names and numeric values agree exactly
    but the retry supplies more units, prefer that more complete extraction.
    No numeric values or decimal points are ever corrected here.
    """
    try:
        result = extract_report(
            state["file_bytes"],
            state["filename"],
            retry=True,
        )

        original_text = state.get("extracted_text", "")
        retry_text = result.get("text", "")
        method = result.get("method", "Alternative OCR strategy")

        if original_text.strip() and retry_text.strip():
            original_results = extract_lab_results(original_text)
            retry_results = extract_lab_results(retry_text)

            def normalized_name(item):
                name = str(item.get("test", item.get("name", ""))).lower()
                name = re.sub(r"[^a-z0-9]+", " ", name)
                name = " ".join(name.split())
                # Common OCR variants of HbA1c should compare as the same test.
                compact = name.replace(" ", "")
                if compact in {"hba1c", "hbalc", "hbaic", "hbaic"}:
                    return "hba1c"
                if compact in {"fastingbloodglucose", "fastingglucose"}:
                    return "fasting blood glucose"
                if compact in {"randombloodglucose", "randomglucose"}:
                    return "random blood glucose"
                return name

            def normalized_value(item):
                # Compare the literal numeric value; never infer or restore a decimal.
                raw = str(item.get("value", "")).replace(",", "").strip()
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
                    return raw
                try:
                    return float(raw)
                except ValueError:
                    return raw

            def result_map(results):
                mapped = {}
                for item in results:
                    name = normalized_name(item)
                    if not name:
                        continue
                    mapped[name] = {
                        "value": normalized_value(item),
                        "unit": str(item.get("unit", "")).strip(),
                    }
                return mapped

            original_map = result_map(original_results)
            retry_map = result_map(retry_results)

            # If identified tests or their numeric values differ, do not choose
            # between competing OCR interpretations automatically.
            values_agree = (
                bool(original_map)
                and original_map.keys() == retry_map.keys()
                and all(
                    original_map[name]["value"] == retry_map[name]["value"]
                    for name in original_map
                )
            )

            if not values_agree:
                return {
                    "extracted_text": original_text,
                    "extraction_method": f"{method} — retry disagreement; original text retained",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "message": (
                        "The initial extraction and OCR retry disagreed about one or more "
                        "test names or numeric values. The original extracted text was retained; "
                        "manual review is required. No values were automatically corrected."
                    ),
                }

            # When numeric values agree, prefer the extraction that explicitly
            # includes more units. This does not change any value or decimal.
            original_units = sum(bool(item["unit"]) for item in original_map.values())
            retry_units = sum(bool(item["unit"]) for item in retry_map.values())

            if retry_units > original_units:
                return {
                    "extracted_text": retry_text,
                    "extraction_method": f"{method} — same values; retry has more unit information",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "message": (
                        "The initial extraction and retry identified the same test names and "
                        "numeric values. The extraction with more explicit unit information "
                        "was selected; all values remain unchanged. Verify the report against "
                        "the original source."
                    ),
                }

            return {
                "extracted_text": original_text,
                "extraction_method": f"{method} — numeric values agree; original text retained",
                "retry_count": state.get("retry_count", 0) + 1,
                "message": result.get("warning") or "Numeric values agree; original extraction retained.",
            }

        # If either attempt has no text, retain the original when available.
        selected_text = original_text or retry_text
        return {
            "extracted_text": selected_text,
            "extraction_method": f"{method} — incomplete extraction; manual review may be required",
            "retry_count": state.get("retry_count", 0) + 1,
            "message": result.get("warning") or "Retry completed, but extraction was incomplete.",
        }

    except Exception as error:
        original_text = state.get("extracted_text", "")
        return {
            "extracted_text": original_text,
            "extraction_method": "Alternative extraction failed — original text retained",
            "retry_count": state.get("retry_count", 0) + 1,
            "message": (
                "Alternative OCR extraction failed. The original extracted text was retained "
                "for review. No values were automatically corrected. "
                f"Technical detail: {error}"
            ),
        }


# --------------------------------------------------
# VALIDATION MESSAGE CLEANUP
# --------------------------------------------------

def clean_validation_message(message):
    """Deduplicate validation notes even when one copy has a Review: prefix."""

    parts = [
        part.strip()
        for part in str(message or "").split(";")
        if part.strip()
    ]

    unique_parts = []
    seen = set()

    for part in parts:
        # A validator may prefix the first note with "Review:" while the
        # graph adds the same note again with that prefix. Compare the note
        # content without the prefix so these copies collapse to one entry.
        content = re.sub(r"^review\s*:\s*", "", part, flags=re.IGNORECASE).strip()
        key = " ".join(content.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique_parts.append(part)

    return "; ".join(unique_parts)


def formatting_checks_passed(results):
    """Check basic formatting without claiming clinical accuracy."""

    if not results:
        return False

    for result in results:
        message = str(
            result.get("validation", "")
        ).strip().lower()

        if not message.startswith("format checks passed"):
            return False

    return True


# --------------------------------------------------
# VALUE HELPERS
# --------------------------------------------------

def normalized_test_name(result):
    """Normalize a laboratory test name."""

    name = result.get("test", result.get("name", ""))

    return " ".join(
        str(name).lower().split()
    )


def numeric_value(result):
    """Convert a numeric result to float when possible."""

    import re

    raw = str(
        result.get("value", "")
    ).replace(",", "").strip()

    if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return None

    try:
        return float(raw)
    except ValueError:
        return None


# --------------------------------------------------
# SUSPICIOUS VALUE DETECTION
# --------------------------------------------------

def flag_suspicious_results(results):
    """
    Flag questionable values for human verification.

    Never change a reported value automatically.
    """

    flagged = False
    updated_results = []

    for original in results:
        result = deepcopy(original)

        test_name = normalized_test_name(result)
        value = numeric_value(result)

        unit = (
            str(result.get("unit", ""))
            .lower()
            .replace(" ", "")
        )

        reasons = []

        # Missing or unreadable numeric values
        if value is None:
            reasons.append(
                "Value is missing or not numeric"
            )

        # HbA1c plausibility checks
        elif "hba1c" in test_name or "hbalc" in test_name:

            if value <= 0 or value > 20:
                reasons.append(
                    "HbA1c value is outside broad plausibility checks"
                )

        # Glucose plausibility checks
        elif "glucose" in test_name:

            if value <= 0 or value > 600:
                reasons.append(
                    "Glucose value is outside broad plausibility checks"
                )

            elif value >= 400:
                reasons.append(
                    "Very high glucose value; verify against the source"
                )

        # Unit checks
        if (
            ("hba1c" in test_name or "hbalc" in test_name)
            and unit
            and unit not in {"%", "percent"}
        ):
            reasons.append(
                "Unexpected HbA1c unit; verify against the source"
            )

        if (
            "glucose" in test_name
            and unit
            and unit not in {
                "mg/dl",
                "mg/dl.",
                "mmol/l",
                "mmol/l.",
            }
        ):
            reasons.append(
                "Unrecognized glucose unit; verify against the source"
            )

        # Add review notes only once
        if reasons:
            flagged = True

            existing = result.get("validation", "")

            review_note = "Review: " + "; ".join(
                dict.fromkeys(reasons)
            )

            result["validation"] = clean_validation_message(
                f"{existing}; {review_note}"
            )

        else:
            result["validation"] = clean_validation_message(
                result.get("validation", "")
            )

        updated_results.append(result)

    return updated_results, flagged


# --------------------------------------------------
# CHECK FOR MISSING TEST RESULTS
# --------------------------------------------------

def expected_tests_missing_from_extraction(text, results):
    """
    Detect supported test names in the extracted text
    that were not represented in the extracted results.
    """

    lower = text.lower()

    extracted_names = [
        normalized_test_name(result)
        for result in results
    ]

    checks = [
        (
            "HbA1c",
            ("hba1c", "hbalc", "hbaic"),
        ),
        (
            "Fasting Blood Glucose",
            ("fasting blood glucose", "fasting glucose"),
        ),
        (
            "Random Blood Glucose",
            ("random blood glucose", "random glucose"),
        ),
    ]

    missing = []

    for display_name, aliases in checks:

        mentioned = any(
            alias in lower for alias in aliases
        )

        represented = any(
            any(alias in name for alias in aliases)
            for name in extracted_names
        )

        if mentioned and not represented:
            missing.append(display_name)

    return missing


# --------------------------------------------------
# VALIDATE RESULTS
# --------------------------------------------------

def validate_results(state: ReportState):
    """Extract and validate laboratory results."""

    text = state.get("extracted_text", "")

    if not text.strip():
        return {
            "lab_results": [],
            "validation_passed": False,
            "status": "processing",
            "message": (
                "No readable report text was extracted. "
                "Another extraction attempt or manual review is required."
            ),
        }

    try:
        results = extract_lab_results(text)

        results = validate_lab_results(results)

        results = [
            deepcopy(result)
            for result in results
        ]

    except Exception as error:
        return {
            "lab_results": [],
            "validation_passed": False,
            "status": "processing",
            "message": (
                f"Result extraction or validation failed: {error}"
            ),
        }

    # Check basic formatting
    passed = formatting_checks_passed(results)

    # Check suspicious values without correcting them
    results, suspicious = flag_suspicious_results(results)

    # Identify named tests that were not extracted
    missing_tests = expected_tests_missing_from_extraction(
        text,
        results,
    )

    if suspicious or missing_tests:
        passed = False

    # Build a clear message
    if not results:

        message = (
            "No supported laboratory results were identified. "
            "Please review the original report."
        )

    elif suspicious and missing_tests:

        message = (
            "Some results look suspicious, and some named tests "
            "were not extracted. Missing tests: "
            + ", ".join(missing_tests)
            + ". Manual review is required. "
            "No values were automatically corrected."
        )

    elif suspicious:

        message = (
            "One or more values look suspicious. Verify the values, "
            "decimal points, and units against the original report. "
            "No values were automatically corrected."
        )

    elif missing_tests:

        message = (
            "The following tests appear in the extracted text "
            "but were not identified as results: "
            + ", ".join(missing_tests)
            + ". Manual review is required."
        )

    elif passed:

        message = (
            "Extraction and basic formatting checks completed. "
            "Verify every test, value, and unit against the original "
            "report. These checks do not establish clinical accuracy."
        )

    else:

        message = (
            "One or more extracted results are uncertain or failed "
            "formatting checks."
        )

    return {
        "lab_results": results,
        "validation_passed": passed,
        "status": "processing",
        "message": message,
    }


# --------------------------------------------------
# ROUTING AFTER INITIAL EXTRACTION
# --------------------------------------------------

def route_after_extraction(state: ReportState):
    """Choose validation, retry, or manual review."""

    text = state.get("extracted_text", "")

    if text.strip():
        return "validate"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"

    return "manual_review"


# --------------------------------------------------
# ROUTING AFTER VALIDATION
# --------------------------------------------------

def route_after_validation(state: ReportState):
    """Retry incomplete extraction; send complete but suspicious results to review.

    Suspicious values alone are not evidence that OCR failed. If all expected
    tests, literal numeric values, and recognized units are present, the
    workflow stops for manual review instead of needlessly running OCR again.
    Missing tests, values, or units can still trigger the configured retry.
    """
    method = str(state.get("extraction_method", "")).lower()
    if "retry disagreement" in method:
        return "manual_review"

    if state.get("validation_passed", False):
        return "complete"

    results = state.get("lab_results", []) or []

    def compact_name(item):
        name = str(item.get("test", item.get("name", ""))).lower()
        return re.sub(r"[^a-z0-9]+", "", name)

    by_name = {compact_name(item): item for item in results if compact_name(item)}
    expected = {
        "hba1c": ("hba1c", "hbalc", "hbaic", "a1c"),
        "fastingbloodglucose": ("fastingbloodglucose", "fastingglucose"),
        "randombloodglucose": ("randombloodglucose", "randomglucose"),
    }

    incomplete = False
    for canonical, aliases in expected.items():
        item = next(
            (value for key, value in by_name.items() if key in aliases),
            None,
        )
        if item is None:
            incomplete = True
            break

        raw_value = str(item.get("value", "")).replace(",", "").strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw_value):
            incomplete = True
            break

        unit = str(item.get("unit", "")).strip().lower().replace(" ", "")
        if canonical == "hba1c":
            unit_ok = unit in {"%", "percent"}
        else:
            unit_ok = unit in {"mg/dl", "mmol/l"}
        if not unit_ok:
            incomplete = True
            break

        validation = str(item.get("validation", "")).lower()
        # The validator uses these messages for content plausibility, not OCR
        # completeness. They should stop for manual review, not trigger another
        # OCR pass when the literal value and expected unit are already present.
        plausibility_warning = any(
            phrase in validation
            for phrase in (
                "outside broad plausibility checks",
                "very high glucose value; verify against the source",
            )
        )
        if "format checks passed" not in validation and not plausibility_warning:
            incomplete = True
            break

    if incomplete and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"

    if incomplete:
        return "manual_review"

    # All expected fields are present and formatted. If validation still failed,
    # it is a content/plausibility issue, not an extraction-completeness issue.
    return "manual_review"


# --------------------------------------------------
# MANUAL REVIEW
# --------------------------------------------------

def manual_review(state: ReportState):
    """Stop automated processing when results remain uncertain."""

    message = state.get("message", "").strip()

    if not message:
        message = (
            "The report could not be validated reliably."
        )

    if "retry disagreement" in str(state.get("extraction_method", "")).lower():
        message = (
            "The initial extraction and OCR retry disagreed. The original extracted text "
            "was retained, and no values were automatically corrected. " + message
        )

    return {
        "status": "manual_review",
        "validation_passed": False,
        "message": (
            "Manual review required. Compare the extracted "
            "text, test names, values, and units with the original "
            "laboratory report. "
            + message
        ),
    }


# --------------------------------------------------
# COMPLETED
# --------------------------------------------------

def complete_report(state: ReportState):
    """Complete processing after the configured checks pass."""

    return {
        "status": "completed",
        "validation_passed": True,
        "message": (
            "Extraction and basic formatting checks completed. "
            "Verify every test, value, and unit against the original "
            "laboratory report. These checks do not establish "
            "clinical accuracy."
        ),
    }


# --------------------------------------------------
# BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

builder = StateGraph(ReportState)

builder.add_node("inspect", inspect_report)
builder.add_node("extract", extract_text)
builder.add_node("retry", retry_with_ocr)
builder.add_node("validate", validate_results)
builder.add_node("manual_review", manual_review)
builder.add_node("complete", complete_report)

builder.add_edge(START, "inspect")
builder.add_edge("inspect", "extract")

builder.add_conditional_edges(
    "extract",
    route_after_extraction,
    {
        "validate": "validate",
        "retry": "retry",
        "manual_review": "manual_review",
    },
)

builder.add_edge("retry", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validation,
    {
        "complete": "complete",
        "retry": "retry",
        "manual_review": "manual_review",
    },
)

builder.add_edge("manual_review", END)
builder.add_edge("complete", END)

report_graph = builder.compile()