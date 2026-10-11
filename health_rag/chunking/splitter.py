from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from health_rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_report(text: str, *, user_id: str, document_id: str, source_name: str, report_date: str | None = None) -> list[Document]:
    """Split report text with paragraph and sentence boundaries while retaining provenance."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        add_start_index=True,
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"user_id": user_id, "document_id": document_id, "source_name": source_name, "report_date": report_date}],
    )
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks
