import json
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from health_rag.config import MODEL_NAME
from health_rag.schemas import MedicalReport


EXTRACTION_RULES = """Extract a medical report, appointment card, or prescription into the supplied schema. Transcribe only what is visible or stated.
Never infer a medication, dose, unit, date, diagnosis, or reference range. Use null/empty lists when absent.
Mark a medication ACTIVE only if the source explicitly says it is current/active; otherwise use UNKNOWN or the explicitly stated status.
Preserve original units and wording. Record a condition and its duration only if the document explicitly documents them.
Set UNKNOWN when a lab flag is not explicitly shown. Add a note for illegible or uncertain content."""


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def extract_report(
    text: str,
    *,
    image_path: Path | None = None,
    image_data: list[tuple[bytes, str]] | None = None,
) -> MedicalReport:
    model = _llm().with_structured_output(MedicalReport)
    if image_path or image_data:
        import base64
        from mimetypes import guess_type

        images = image_data or [(image_path.read_bytes(), guess_type(image_path.name)[0] or "image/jpeg")]
        content = [{"type": "text", "text": EXTRACTION_RULES}]
        if text.strip():
            content.append({"type": "text", "text": f"Selectable text from the same report:\n{text}"})
        for page_data, mime_type in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(page_data).decode('ascii')}"}})
        message = HumanMessage(content=content)
        raw = model.invoke([message])
    else:
        raw = model.invoke(f"{EXTRACTION_RULES}\n\nREPORT TEXT:\n{text}")
    try:
        if isinstance(raw, MedicalReport):
            return raw
        if isinstance(raw, dict):
            return MedicalReport.model_validate(raw)
        return MedicalReport.model_validate_json(str(raw))
    except ValidationError as exc:
        raise ValueError(f"Structured medical extraction failed validation: {exc}") from exc


def serialize_report(report: MedicalReport) -> str:
    return json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
