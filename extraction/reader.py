"""PDF/image text extraction with OCR alternatives that preserve punctuation.

OCR output is never numerically auto-corrected. Suspicious values are passed to
validation so the workflow can request manual review.
"""

import io
import os
import re

import cv2
import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from pytesseract import Output


# Optional environment overrides make this portable between Windows machines.
TESSERACT_PATH = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
TESSDATA_PATH = os.getenv(
    "TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata"
)
if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
if os.path.isdir(TESSDATA_PATH):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


TEST_PATTERNS = [
    r"\bhba1c\b", r"\bhbalc\b", r"\ba1c\b",
    r"\bfasting\b", r"\brandom\b", r"\bglucose\b", r"\bblood sugar\b",
]


def extract_pdf_text(file_bytes):
    """Extract selectable text from a PDF."""
    document = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text("text") for page in document).strip()
    finally:
        document.close()


def deskew_image(image):
    """Correct modest scan rotation without aggressively cleaning small dots."""
    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    coordinates = np.column_stack(np.where(threshold > 0))
    if len(coordinates) < 100:
        return image

    angle = cv2.minAreaRect(coordinates.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return image

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image_array, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)


def preprocess_image(image, retry=False):
    """Upscale and enhance legibility; deliberately avoid hard thresholding."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1800:
        scale = 1800 / image.width
        image = image.resize(
            (1800, max(1, int(image.height * scale))), Image.Resampling.LANCZOS
        )
    image = deskew_image(image)
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.8 if not retry else 2.0)
    image = ImageEnhance.Sharpness(image).enhance(1.6 if not retry else 2.0)
    # Do not binarize or median-filter: those steps can erase decimal-point pixels.
    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))


def _image_variants(image, retry=False):
    """Return full-page and result-table-focused variants without erasing decimal dots.

    Laboratory tables are often small relative to the page. OCRing the lower
    portion separately gives the result rows more pixels and can recover units
    such as mg/dL that are missed during full-page OCR.
    """
    base = preprocess_image(image, retry=retry)
    variants = [base, ImageEnhance.Contrast(base).enhance(1.15)]
    if retry:
        variants.append(ImageEnhance.Brightness(base).enhance(1.08))

    # Include a generous lower-page crop; avoid relying on exact page layout.
    # It is an OCR candidate only, not a source for filling missing values.
    if base.height >= 500:
        top = int(base.height * 0.28)
        table_crop = base.crop((0, top, base.width, base.height))
        if table_crop.width < 1800:
            scale = 1800 / table_crop.width
            table_crop = table_crop.resize(
                (1800, max(1, int(table_crop.height * scale))), Image.Resampling.LANCZOS
            )
        variants.append(table_crop)

    # Add a table-focused perspective-normalized candidate. The regular full-page
    # candidates remain in the list and retain priority when they score better.
    try:
        table_variant = _rectified_table_variant(image)
        if table_variant is not None:
            variants.append(table_variant)
    except (cv2.error, ValueError):
        pass
    return variants



def _rectified_table_variant(image):
    """Create an extra OCR candidate from a perspective-normalized result-table region.

    This is an additional candidate only; it does not supply or modify any values.
    The relative region covers the central results-table area on common report scans.
    """
    rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    height, width = rgb.shape[:2]
    if width < 400 or height < 500:
        return None

    # Broad proportional crop around the results table; keep a little margin
    # so the table's outer borders remain available for perspective correction.
    x1, x2 = int(width * 0.065), int(width * 0.94)
    y1, y2 = int(height * 0.295), int(height * 0.49)
    region = rgb[y1:y2, x1:x2]
    if region.size == 0:
        return None

    # The table edges are slightly skewed in the supplied scan. Normalize that
    # skew with a conservative quadrilateral based on the crop boundaries.
    rh, rw = region.shape[:2]
    source = np.float32([
        [0, rh * 0.02], [rw - 1, 0],
        [rw - 1, rh - 1], [0, rh - 1],
    ])
    target_width = max(1200, rw)
    target_height = max(340, int(target_width * rh / rw))
    destination = np.float32([
        [0, 0], [target_width - 1, 0],
        [target_width - 1, target_height - 1], [0, target_height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(source, destination)
    normalized = cv2.warpPerspective(
        region, matrix, (target_width, target_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(normalized)

def score_ocr_text(text):
    """Score OCR candidates by complete result rows, not report headings alone.

    This only selects among OCR outputs; it never changes extracted numbers or
    supplies a missing unit from a reference range.
    """
    lines = [re.sub(r"\s+", " ", line).strip().lower() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    lower = "\n".join(nonempty)
    score = min(len(text.strip()), 300) / 30

    if re.search(r"\b(laboratory|lab report|diagnostic|specimen|test results?)\b", lower):
        score += 15

    test_groups = {
        "hba1c": re.compile(r"\b(?:hba1c|hbalc|hbaic|hb[a4]1c|a1c)\b"),
        "fasting": re.compile(r"\bfasting\b.{0,35}\bglucose\b|\bglucose\b.{0,35}\bfasting\b"),
        "random": re.compile(r"\brandom\b.{0,35}\bglucose\b|\bglucose\b.{0,35}\brandom\b"),
    }
    number_re = re.compile(r"(?<![\d.])\d+(?:[.,]\d+)?(?![\d.])")
    unit_re = re.compile(r"%|mg\s*/\s*d[l1i]|mmol\s*/\s*[l1i]")

    # Table OCR may place the test name, value, and unit on adjacent lines.
    # Evaluate a small local window but don't infer that a unit belongs to a
    # result; this is only candidate-ranking, not extraction/validation.
    found_tests = 0
    complete_rows = 0
    for i, line in enumerate(nonempty):
        matched = [name for name, pattern in test_groups.items() if pattern.search(line)]
        if not matched:
            continue
        found_tests += len(matched)
        window = " ".join(nonempty[max(0, i-1):min(len(nonempty), i+3)])
        has_number = bool(number_re.search(window))
        has_unit = bool(unit_re.search(window))
        if has_number:
            score += 45
        if has_number and has_unit:
            score += 55
            complete_rows += 1

    # Strongly prefer candidates containing all three named test results.
    score += found_tests * 12
    score += complete_rows * 18
    if found_tests < 3:
        score -= (3 - found_tests) * 20
    if re.search(r"\b\d+(?:[.,]\d+)?\s*%", lower):
        score += 8
    if re.search(r"\b\d+(?:[.,]\d+)?\s*mg\s*/\s*d[l1i]\b", lower):
        score += 8

    if re.search(r"\b(sales receipt|receipt no|item qty|amount \(pkr\)|supermarket)\b", lower):
        score -= 300
    return score


def extract_positioned_text(image, config):
    """Group Tesseract words into lines using their coordinates."""
    data = pytesseract.image_to_data(
        image, lang="eng", config=config, output_type=Output.DICT
    )
    words = []
    for index, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1
        if confidence < 0:
            continue
        left, top = int(data["left"][index]), int(data["top"][index])
        height = int(data["height"][index])
        words.append({"text": word, "x": left, "y": top + height // 2, "height": height})

    if not words:
        return ""
    words.sort(key=lambda item: (item["y"], item["x"]))
    rows = []
    for word in words:
        best_row, best_distance = None, None
        for row in rows:
            row_y = sum(item["y"] for item in row) / len(row)
            distance = abs(word["y"] - row_y)
            if distance <= max(12, word["height"]):
                if best_distance is None or distance < best_distance:
                    best_row, best_distance = row, distance
        (rows.append([word]) if best_row is None else best_row.append(word))

    return "\n".join(
        " ".join(item["text"] for item in sorted(row, key=lambda item: item["x"]))
        for row in rows
    )


def _merge_result_lines(candidates):
    """Prefer the best OCR line for each supported test across actual OCR passes.

    This is not value correction: a value/unit is used only if Tesseract literally
    returned it in one of the candidates. Other report text is retained from the
    highest-scoring full candidate.
    """
    if not candidates:
        return ""
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    merged_text = candidates[0][1]
    lines_by_test = {
        "hba1c": re.compile(r"\b(?:hba1c|hbalc|hbaic|hb[a4]1c|a1c)\b", re.I),
        "fasting": re.compile(r"\bfasting\b.{0,45}\bglucose\b|\bglucose\b.{0,45}\bfasting\b", re.I),
        "random": re.compile(r"\brandom\b.{0,45}\bglucose\b|\bglucose\b.{0,45}\brandom\b", re.I),
    }
    number_re = re.compile(r"(?<![\d.])\d+(?:[.,]\d+)?(?![\d.])")
    unit_re = re.compile(r"%|mg\s*/\s*d[l1i]|mmol\s*/\s*[l1i]", re.I)
    base_lines = merged_text.splitlines()
    normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in base_lines]

    for test_name, test_pattern in lines_by_test.items():
        best_line = None
        best_line_score = -1
        for candidate_score, candidate_text, *_ in candidates:
            lines = [re.sub(r"\s+", " ", line).strip() for line in candidate_text.splitlines() if line.strip()]
            for idx, line in enumerate(lines):
                if not test_pattern.search(line):
                    continue
                # Check only a narrow row window for OCR that splits table cells.
                window = " ".join(lines[idx:idx + 2])
                has_number = bool(number_re.search(window))
                has_unit = bool(unit_re.search(window))
                line_score = candidate_score + (25 if has_number else 0) + (60 if has_unit else 0)
                if line_score > best_line_score:
                    best_line_score, best_line = line_score, line
        if best_line:
            # Replace only the matching test line in the selected full text.
            target_index = next((i for i, line in enumerate(normalized_lines) if test_pattern.search(line)), None)
            if target_index is not None:
                normalized_lines[target_index] = best_line
            else:
                normalized_lines.append(best_line)
    return "\n".join(normalized_lines).strip()


def extract_image_text(image, debug=False, retry=False):
    """Try several OCR segmentations and keep the most report-like candidate."""
    configurations = (
        ["--oem 3 --psm 4", "--oem 3 --psm 6", "--oem 3 --psm 11"]
        if retry else
        ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 11", "--oem 3 --psm 12"]
    )
    candidates = []
    for variant_index, variant in enumerate(_image_variants(image, retry=retry)):
        for config in configurations:
            try:
                regular = pytesseract.image_to_string(
                    variant, lang="eng", config=config
                ).strip()
                if regular:
                    candidates.append((score_ocr_text(regular), regular, config, "standard", variant_index))
                positioned = extract_positioned_text(variant, config).strip()
                if positioned:
                    candidates.append((score_ocr_text(positioned), positioned, config, "positioned", variant_index))
            except pytesseract.TesseractError as error:
                if debug:
                    print(f"Tesseract configuration failed ({config}): {error}")

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_text, best_config, best_kind, best_variant = candidates[0]
    best_text = _merge_result_lines(candidates)
    if debug:
        print("\n--- OCR CANDIDATE SUMMARY ---")
        for score, text, config, kind, variant_index in candidates[:6]:
            print(f"score={score:.1f}, config={config}, kind={kind}, variant={variant_index}")
            print(text[:1000])
        print(f"--- SELECTED OCR: {best_config}, {best_kind}, variant={best_variant} ---")
    return best_text


def _has_complete_lab_results(text):
    """Return True only when all three supported tests have literal result values.

    A PDF may contain selectable headings and table labels but no extractable
    table cells. Do not treat that partial text as a successful extraction.
    This check does not alter values or infer units.
    """
    if not text or not text.strip():
        return False

    normalized = re.sub(r"\s+", " ", text.lower())
    patterns = {
        "hba1c": r"\b(?:hb\s*a\s*1\s*c|hba1c|hbalc|hbaic|a1c)\b",
        "fasting": r"\bfasting\b.{0,45}\bglucose\b|\bglucose\b.{0,45}\bfasting\b",
        "random": r"\brandom\b.{0,45}\bglucose\b|\bglucose\b.{0,45}\brandom\b",
    }
    # Check the local text after each test label for a literal numeric token.
    for pattern in patterns.values():
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            return False
        nearby = normalized[match.end():match.end() + 90]
        if not re.search(r"(?<![\d.])\d{1,4}(?:[.,]\d+)?(?![\d.])", nearby):
            return False
    return True


def extract_report(file_bytes, filename, retry=False):
    """Extract text from selectable/scanned PDFs or common image formats."""
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        if not retry:
            text = extract_pdf_text(file_bytes)
            # Trust selectable text only when it contains actual values for all
            # supported tests. Headings/table headers alone must trigger OCR.
            if _has_complete_lab_results(text):
                return {"text": text, "method": "PDF text extraction", "warning": None}

        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = []
        try:
            for page in document:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                pages.append(extract_image_text(image, debug=False, retry=retry))
        finally:
            document.close()
        text = "\n".join(pages).strip()
        method = "OCR with enhanced preprocessing (retry)" if retry else "OCR for scanned PDF"
        return {"text": text, "method": method, "warning": None if text else "OCR could not extract readable text."}

    with Image.open(io.BytesIO(file_bytes)) as image:
        text = extract_image_text(image, debug=False, retry=retry)
    method = "Image OCR with enhanced preprocessing (retry)" if retry else "Image OCR"
    return {"text": text, "method": method, "warning": None if text else "OCR could not extract readable text."}
