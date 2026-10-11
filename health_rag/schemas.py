from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class LabResult(BaseModel):
    test_name: str
    value: str = Field(description="Value exactly as shown, including inequality signs")
    unit: str | None = None
    reference_range: str | None = None
    flag: Literal["NORMAL", "HIGH", "LOW", "ABNORMAL", "UNKNOWN"] = "UNKNOWN"


class Medication(BaseModel):
    drug_name: str
    dosage: str | None = Field(default=None, description="Only if explicitly present")
    frequency: str | None = None
    route: str | None = None
    status: Literal["ACTIVE", "DISCONTINUED", "AS_NEEDED", "UNKNOWN"] = "UNKNOWN"
    confidence: float = Field(ge=0, le=1)


class Diagnosis(BaseModel):
    condition: str
    diagnosis_date: date | None = None
    duration_as_documented: str | None = Field(default=None, description="Only verbatim/documented duration; never calculate or infer")
    status: str | None = None


class MedicalReport(BaseModel):
    report_type: str = Field(description="For example blood test, colonoscopy, prescription")
    patient_name: str | None = None
    report_date: date | None = None
    clinician: str | None = None
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


class UploadResult(BaseModel):
    user_id: str
    document_id: str
    source_name: str
    report: MedicalReport
    chunk_count: int


class RetrievedChunk(BaseModel):
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    similarity: float
