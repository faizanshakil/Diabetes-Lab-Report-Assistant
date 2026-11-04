"""End-to-end LangGraph integration smoke tests.

Run from the project root after installing requirements:
    python -m unittest tests.test_langgraph_integration -v

These tests use the real compiled LangGraph graph, while stubbing only the
file extractor so they can verify node routing deterministically without OCR
runtime variability.
"""
import unittest
from unittest.mock import patch

from workflow.graph import report_graph
import workflow.graph as graph_module
from validation.result_validator import extract_lab_results


NORMAL_TEXT = """EXAMPLE DIAGNOSTIC LABORATORY
TRAINING SAMPLE — NOT A REAL PATIENT REPORT
LABORATORY RESULTS
Fasting Blood Glucose 108 mg/dL 70-99
HbA1c 5.8% Below 5.7
Random Blood Glucose 132 mg/dL Lab-specific
"""

SUSPICIOUS_TEXT = """EXAMPLE DIAGNOSTIC LABORATORY
SYNTHETIC TEST DOCUMENT — NOT A REAL PATIENT REPORT
LABORATORY RESULTS
HbA1c: 58% | Reference range: typically below 5.7%
Fasting Blood Glucose: 1080 mg/dL | Reference range: 70-99 mg/dL
Random Blood Glucose: 132 mg/dL | Lab-specific; often below 140 mg/dL
Software test note: Two results above are intentionally suspicious for validation testing.
"""


def invoke_with_extractor(first_result, retry_result=None, filename="test.pdf"):
    """Invoke the real graph while controlling extraction outputs."""
    calls = {"count": 0}

    def fake_extract_report(file_bytes, received_filename, retry=False):
        calls["count"] += 1
        if retry:
            return retry_result or {"text": "", "method": "Mock retry"}
        return first_result

    initial_state = {
        "file_bytes": b"synthetic-test-file",
        "filename": filename,
        "extracted_text": "",
        "extraction_method": "",
        "lab_results": [],
        "validation_passed": False,
        "retry_count": 0,
        "status": "processing",
        "message": "",
    }

    with patch.object(graph_module, "extract_report", side_effect=fake_extract_report):
        final_state = report_graph.invoke(initial_state)

    return final_state, calls["count"]


class LangGraphIntegrationTests(unittest.TestCase):
    def test_normal_report_reaches_completed(self):
        state, calls = invoke_with_extractor({
            "text": NORMAL_TEXT,
            "method": "Mock PDF text extraction",
        })
        self.assertEqual(state["status"], "completed", state)
        self.assertTrue(state["validation_passed"], state)
        found = {row["test"]: row for row in state["lab_results"]}
        self.assertIn("HbA1c", found)
        self.assertIn("Fasting Blood Glucose", found)
        self.assertIn("Random Blood Glucose", found)
        self.assertEqual(calls, 1)

    def test_suspicious_report_routes_to_manual_review_and_preserves_values(self):
        state, calls = invoke_with_extractor({
            "text": SUSPICIOUS_TEXT,
            "method": "Mock PDF text extraction",
        })
        self.assertEqual(state["status"], "manual_review", state)
        self.assertFalse(state["validation_passed"], state)
        found = {row["test"]: row for row in state["lab_results"]}
        self.assertEqual(found["HbA1c"]["value"], 58.0)
        self.assertEqual(found["Fasting Blood Glucose"]["value"], 1080.0)
        self.assertEqual(found["Random Blood Glucose"]["value"], 132.0)
        self.assertEqual(calls, 1)

    def test_unreadable_report_retries_once_then_manual_review(self):
        state, calls = invoke_with_extractor(
            {"text": "", "method": "Mock initial OCR"},
            {"text": "", "method": "Mock retry OCR"},
            filename="unreadable.png",
        )
        self.assertEqual(state["status"], "manual_review", state)
        self.assertEqual(state["retry_count"], 1, state)
        self.assertEqual(calls, 2)

    def test_retry_disagreement_routes_to_manual_review(self):
        # The normal report would correctly complete without a retry. Use a
        # first extraction with a missing required result so the real graph
        # routes through retry, then return a conflicting numeric value.
        incomplete_text = NORMAL_TEXT.replace(
            "Random Blood Glucose 132 mg/dL Lab-specific",
            "Random Blood Glucose [unreadable] mg/dL Lab-specific",
    )
        retry_text = NORMAL_TEXT.replace("108 mg/dL", "180 mg/dL")
        state, calls = invoke_with_extractor(
            {"text": incomplete_text, "method": "Mock initial OCR"},
            {"text": retry_text, "method": "Mock retry OCR"},
            filename="disagreement.png",
        )
        self.assertEqual(state["status"], "manual_review", state)
        self.assertIn("retry disagreement", state["extraction_method"].lower())
        self.assertEqual(calls, 2)
        # A disagreement must retain the original extraction, not the retry.
        self.assertEqual(state["extracted_text"], incomplete_text)


class ReportMetadataExtractionTests(unittest.TestCase):
    def test_extracts_explicit_report_date_and_reference_ranges(self):
        text = """EXAMPLE DIAGNOSTIC LABORATORY
Report Date: 04/11/2026
LABORATORY RESULTS
Fasting Blood Glucose: 108 mg/dL | Reference range: 70-99 mg/dL
HbA1c: 5.8% | Reference range: below 5.7%
Random Blood Glucose: 132 mg/dL | Reference range: Not provided
"""
        results = {row["test"]: row for row in extract_lab_results(text)}
        self.assertEqual(results["HbA1c"]["report_date"], "04/11/2026")
        self.assertEqual(results["Fasting Blood Glucose"]["reference_range"], "70-99 mg/dL")
        self.assertEqual(results["HbA1c"]["reference_range"], "below 5.7%")

    def test_missing_metadata_is_not_invented(self):
        text = """LABORATORY RESULTS
Fasting Blood Glucose 108 mg/dL
HbA1c 5.8%
Random Blood Glucose 132 mg/dL
"""
        results = extract_lab_results(text)
        self.assertTrue(results)
        self.assertTrue(all(row["report_date"] == "Not identified" for row in results))
        self.assertTrue(all(row["reference_range"] == "Not identified" for row in results))


if __name__ == "__main__":
    unittest.main(verbosity=2)
