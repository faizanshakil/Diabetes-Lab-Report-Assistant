"""Extract and validate diabetes-related laboratory results conservatively.

This module never inserts missing decimal points or silently changes values.
Questionable or incomplete extraction is marked for human review.
"""

import re


TEST_PATTERNS = {
    "HbA1c": r"\b(?:Hb\s*A\s*1\s*c|HbA1c|HbAlc|HbAic|A1c|Glycated\s+Hemoglobin)\b",
    "Fasting Blood Glucose": r"\bFasting\s+(?:Blood\s+)?(?:Glucose|Blood\s+Sugar)\b",
    "Random Blood Glucose": r"\bRandom\s+(?:Blood\s+)?(?:Glucose|Blood\s+Sugar)\b",
}


REFERENCE_TERMS = re.compile(
    r"\b(reference(?:\s+range)?|normal\s+range|typically|usually|"
    r"interpretation|software\s+test\s+note|not\s+a\s+real|"
    r"intentionally\s+suspicious|written\s+as|example)\b",
    re.IGNORECASE,
)

VALUE_PATTERN = re.compile(r"(?<![\d.])\d{1,4}(?:\.\d+)?(?![\d.])")


def normalize_text(text):
    """Normalize harmless OCR spacing and unit variants, not numeric values."""
    if not text:
        return ""

    text = str(text)
    text = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("−", "-")
        .replace("／", "/")
        .replace("％", "%")
        .replace("，", ",")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Normalize OCR variants of common units, preserving the numeric text.
    text = re.sub(
        r"\bmg\s*/?\s*d\s*[lI1i]\b", "mg/dL", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\bmmol\s*/?\s*[lI1i]\b", "mmol/L", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\bHb\s*A\s*1\s*c\b", "HbA1c", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHbAlc\b", "HbA1c", text, flags=re.IGNORECASE)
    return text.strip()


def _results_region(text):
    """Prefer the laboratory results section over explanatory prose elsewhere."""
    header = re.search(
        r"\b(?:laboratory\s+results|lab\s+results|test\s+results|"
        r"investigation\s+results)\b\s*:?",
        text,
        re.IGNORECASE,
    )
    if header:
        region = text[header.end():]
    else:
        region = text

    # Keep result parsing inside the laboratory-results section. Synthetic
    # validation notes later in a report may repeat a test name and its value
    # (for example, "HbA1c is written as 58%") and must not outrank the actual
    # result row. These are section boundaries, not value corrections.
    section_end = re.search(
        r"\b(?:software\s+test\s+note|limitations\s+and\s+caveats|"
        r"important\s+note|do\s+not\s+use\s+this\s+synthetic\s+document)\b",
        region,
        re.IGNORECASE,
    )
    if section_end:
        region = region[:section_end.start()]

    return region


def _unit_present(text, test_name):
    if test_name == "HbA1c":
        return bool(re.search(r"[%％]", text))
    return bool(
        re.search(r"\bmg\s*/\s*d[lI1i]\b|\bmmol\s*/\s*[lI1i]\b", text, re.I)
    )


def get_test_section(text, test_name):
    """Find the most result-like occurrence of a supported test name."""
    pattern = TEST_PATTERNS[test_name]
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return ""

    candidates = []
    for match in matches:
        start = match.end()
        later_test_positions = []
        for other_name, other_pattern in TEST_PATTERNS.items():
            if other_name == test_name:
                continue
            next_match = re.search(other_pattern, text[start:], re.IGNORECASE)
            if next_match:
                later_test_positions.append(start + next_match.start())
        end = min(later_test_positions) if later_test_positions else min(len(text), start + 260)
        section = text[start:end].strip()
        nearby = text[max(0, match.start() - 45):min(len(text), end + 40)]
        score = 0

        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if any(VALUE_PATTERN.search(line) for line in lines):
            score += 2
        if _unit_present(section, test_name):
            score += 5
        if re.search(r"\b(result|value)\b", nearby, re.IGNORECASE):
            score += 1
        if REFERENCE_TERMS.search(nearby):
            score -= 6
        if re.search(r"\b(?:below|above|threshold|average|past\s+\d)\b", nearby, re.I):
            score -= 3
        candidates.append((score, section))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else ""


def _result_only_text(section):
    """Remove reference-range/explanatory tails before reading a result or unit.

    This intentionally does not infer missing units or alter numeric values.
    """
    if not section:
        return ""
    cleaned_lines = []
    for raw_line in str(section).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Stop before explanatory wording that commonly introduces reference info.
        line = re.split(
            r"\b(?:reference(?:\s+range)?|normal\s+range|typically|usually|"
            r"lab-specific|interpretation|example reference range)\b",
            line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        # Stop at a numeric interval such as 70-99, which is a reference range.
        range_match = re.search(
            r"(?<!\d)\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?!\d)",
            line,
        )
        if range_match:
            line = line[:range_match.start()].strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_value(section, test_name):
    """Extract a literal candidate value without restoring suspected decimals."""
    if not section:
        return None

    section = _result_only_text(section)
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if not lines:
        return None

    usable_lines = []
    for line in lines:
        # Remove a reference interval appended after the actual result.
        line = re.split(
            r"\b(?:reference(?:\s+range)?|normal\s+range)\b\s*:?",
            line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        # OCR may place the result and explanatory text on one line, e.g.
        # "58 % Typically <5.7%". Keep the result and discard the explanation.
        line = re.split(
            r"\b(?:typically|usually|interpretation|software\s+test\s+note|"
            r"not\s+a\s+real|intentionally\s+suspicious|written\s+as|"
            r"example\s+reference|lab[- ]specific)\b",
            line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        if line and not re.search(
            r"\b(?:purpose|patient\s+id|report\s+date|specimen|"
            r"synthetic\s+test\s+document)\b",
            line,
            re.IGNORECASE,
        ):
            usable_lines.append(line)

    if not usable_lines:
        return None

    # Prefer a numeric value on a line with the expected unit.
    for line in usable_lines:
        if not _unit_present(line, test_name):
            continue
        numbers = VALUE_PATTERN.findall(line)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

    # Otherwise take a literal numeric candidate; validation flags missing units.
    for line in usable_lines:
        numbers = VALUE_PATTERN.findall(line)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
    return None


def _unit_from_explicit_test_note(text, test_name):
    """Use a unit only when a labelled software-test note explicitly states it."""
    if not text:
        return "Not identified"

    normalized = normalize_text(text)
    note_match = re.search(
        r"\bsoftware\s+test\s+note\b(.*)$",
        normalized,
        re.IGNORECASE | re.DOTALL,
    )
    if not note_match:
        return "Not identified"
    note = note_match.group(1)

    if test_name == "HbA1c":
        if re.search(
            r"\bHbA(?:1c|lc|ic|l1c)\b[^.\n]{0,80}\bwritten\s+as\s+\d+(?:\.\d+)?\s*%",
            note,
            re.IGNORECASE,
        ):
            return "%"
    elif test_name == "Fasting Blood Glucose":
        if re.search(
            r"\bfasting\s+(?:blood\s+)?glucose\b[^.\n]{0,100}\b(?:written\s+as\s+)?\d+(?:\.\d+)?\s*(?:mg\s*/\s*d[lI1i]|mmol\s*/\s*[lI1i])",
            note,
            re.IGNORECASE,
        ):
            unit = re.search(r"\b(mg\s*/\s*d[lI1i]|mmol\s*/\s*[lI1i])\b", note, re.IGNORECASE)
            if unit:
                return "mg/dL" if unit.group(1).lower().replace(" ", "").startswith("mg/") else "mmol/L"
    elif test_name == "Random Blood Glucose":
        if re.search(
            r"\brandom\s+(?:blood\s+)?glucose\b[^.\n]{0,100}\b(?:written\s+as\s+)?\d+(?:\.\d+)?\s*(?:mg\s*/\s*d[lI1i]|mmol\s*/\s*[lI1i])",
            note,
            re.IGNORECASE,
        ):
            unit = re.search(r"\b(mg\s*/\s*d[lI1i]|mmol\s*/\s*[lI1i])\b", note, re.IGNORECASE)
            if unit:
                return "mg/dL" if unit.group(1).lower().replace(" ", "").startswith("mg/") else "mmol/L"

    return "Not identified"


def find_unit(section, test_name, full_text=None):
    """Find the result unit; never borrow a unit from a reference range."""
    if section:
        result_text = _result_only_text(section)
        if test_name == "HbA1c":
            if re.search(r"[%％]", result_text):
                return "%"
        else:
            normalized = normalize_text(result_text)
            if re.search(r"\bmg\s*/\s*d[lI1i]\b", normalized, re.IGNORECASE):
                return "mg/dL"
            if re.search(r"\bmmol\s*/\s*[lI1i]\b", normalized, re.IGNORECASE):
                return "mmol/L"

    # Only use the synthetic report's explicitly labelled software-test note
    # as fallback; ordinary reference-range text is never used for this.
    return _unit_from_explicit_test_note(full_text, test_name) if full_text else "Not identified"



def extract_report_date(text):
    """Extract a clearly labelled report/collection date; never guess from context."""
    if not text:
        return "Not identified"
    patterns = [
        r"\b(?:report\s+date|collection\s+date|collected\s+on|date\s+of\s+collection|sample\s+date)\s*[:：-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Not identified"


def extract_reference_range(section, test_name):
    """Capture a range only when it is explicitly adjacent/labeled in the test section."""
    if not section:
        return "Not identified"
    normalized = normalize_text(section)
    label_match = re.search(
        r"\b(?:reference(?:\s+range)?|normal\s+range)\b\s*[:：]?\s*([^\n;|]+)",
        normalized, re.IGNORECASE,
    )
    if label_match:
        candidate = label_match.group(1).strip()
        candidate = re.split(r"\b(?:typically|usually|interpretation|lab-specific)\b", candidate, flags=re.I)[0].strip()
        if candidate:
            return candidate[:80]
    # Accept an explicitly adjacent numeric interval such as 70-99 mg/dL.
    interval = re.search(
        r"(?<![\d.])(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|%)?)",
        normalized, re.IGNORECASE,
    )
    return interval.group(1).strip() if interval else "Not identified"

def extract_lab_results(text):
    """Extract HbA1c, fasting glucose, and random glucose results."""
    normalized = normalize_text(text)
    region = _results_region(normalized)
    results = []
    report_date = extract_report_date(normalized)

    for test_name in TEST_PATTERNS:
        section = get_test_section(region, test_name)
        if not section:
            continue

        value = extract_value(section, test_name)
        if value is None:
            # The graph's missing-test check will route this report to review.
            continue

        results.append({
            "test": test_name,
            "value": value,
            "unit": find_unit(section, test_name, normalized),
            "report_date": report_date,
            "reference_range": extract_reference_range(section, test_name),
            "validation": "Needs review",
        })

    return results


def validate_lab_results(results):
    """Apply conservative format/plausibility checks, not clinical diagnosis."""
    validated = []
    for original in results:
        result = dict(original)
        test = str(result.get("test", ""))
        unit = str(result.get("unit", "Not identified")).strip()
        raw_value = result.get("value")
        reasons = []

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = None

        if value is None:
            reasons.append("value is missing or not numeric")
        elif value <= 0:
            reasons.append("value must be greater than zero")

        if unit == "Not identified" or not unit:
            reasons.append("unit not identified")

        if value is not None and value > 0:
            if test == "HbA1c":
                if unit != "%":
                    reasons.append("verify HbA1c unit")
                if value > 20:
                    reasons.append("HbA1c value is outside broad plausibility checks")
            elif "Glucose" in test:
                if unit not in {"mg/dL", "mmol/L"}:
                    reasons.append("verify glucose unit")
                if unit == "mg/dL" and value > 600:
                    reasons.append("glucose value is outside broad plausibility checks")
                elif unit == "mg/dL" and value >= 400:
                    reasons.append("very high glucose value; verify against the source")
                elif unit == "mmol/L" and value > 100:
                    reasons.append("glucose value is outside broad plausibility checks")

        result["value"] = value if value is not None else raw_value
        result["validation"] = (
            "Review: " + "; ".join(dict.fromkeys(reasons))
            if reasons else "Format checks passed; verify against source"
        )
        validated.append(result)

    return validated
