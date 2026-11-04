from typing import TypedDict, List, Dict, Any


class ReportState(TypedDict):
    file_bytes: bytes
    filename: str
    extracted_text: str
    extraction_method: str
    lab_results: List[Dict[str, Any]]
    validation_passed: bool
    retry_count: int
    status: str
    message: str