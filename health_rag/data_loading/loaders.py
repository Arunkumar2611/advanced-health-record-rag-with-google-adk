from pathlib import Path

from docx import Document as WordDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}


def load_text(path: Path) -> str:
    """Load text from a supported report; image OCR is handled by Gemini separately."""
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or '(no extension)'}")
    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if extension == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if extension == ".docx":
        document = WordDocument(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    return ""
