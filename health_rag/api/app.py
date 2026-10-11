from pathlib import Path
from uuid import uuid4
import asyncio
import zipfile
from io import BytesIO

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from health_rag.api.models import ChatRequest, RegisterRequest
from health_rag.chunking.splitter import chunk_report
from health_rag.config import DATA_DIR, MAX_UPLOAD_BYTES
from health_rag.data_loading.loaders import SUPPORTED_EXTENSIONS, load_text
from health_rag.data_loading.ocr import render_scanned_pdf
from health_rag.extraction import extract_report, serialize_report
from health_rag.graph import build_query_graph
from health_rag.observability import emit
from health_rag.persistence.database import Database
from health_rag.security.auth import create_access_token, current_user, hash_password, verify_password
from health_rag.vector_store.faiss_store import UserFaissStore


app = FastAPI(title="Personal Health RAG API", version="0.2.0", description="Authenticated, report-grounded health information POC")
database = Database()
vector_store = UserFaissStore(database)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ALLOWED_MIME = {"application/pdf", "text/plain", "text/markdown", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "image/png", "image/jpeg", "image/webp"}


def validate_file_signature(extension: str, content: bytes) -> None:
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise HTTPException(415, "File contents do not match a PDF")
    if extension == ".docx":
        if not content.startswith(b"PK") or not zipfile.is_zipfile(BytesIO(content)):
            raise HTTPException(415, "File contents do not match a DOCX document")
    if extension in {".png", ".jpg", ".jpeg", ".webp"}:
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise HTTPException(415, "Unsupported image contents")
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(415, "Uploaded file is not a valid supported image") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> dict:
    if len(payload.password) < 12:
        raise HTTPException(400, "Password must contain at least 12 characters")
    user_id = str(uuid4())
    try:
        database.create_user(user_id=user_id, email=str(payload.email), full_name=payload.full_name.strip(), password_hash=hash_password(payload.password))
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(409, "An account with this email already exists") from exc
        raise
    emit("auth.account_created", user_id=user_id)
    return {"user_id": user_id, "email": str(payload.email).lower(), "full_name": payload.full_name.strip()}


@app.post("/auth/token")
def login(form: OAuth2PasswordRequestForm = Depends()) -> dict:
    user = database.user_by_email(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        emit("auth.login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    emit("auth.login_succeeded", user_id=user["id"])
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "expires_in": 28800}


@app.get("/users/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {key: user[key] for key in ("id", "email", "full_name", "created_at")}


@app.get("/documents")
def documents(user: dict = Depends(current_user)) -> list[dict]:
    return database.list_documents(user["id"])


@app.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_report(file: UploadFile = File(...), user: dict = Depends(current_user)) -> dict:
    user_id = user["id"]
    source_name = Path(file.filename or "upload").name
    extension = Path(source_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {extension or 'unknown'}")
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, "Unsupported content type")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    validate_file_signature(extension, content)

    document_id = str(uuid4())
    user_dir = UPLOAD_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    stored_path = user_dir / f"{document_id}{extension}"
    stored_path.write_bytes(content)
    emit("upload.received", step=1, user_id=user_id, document_id=document_id, source_name=source_name, bytes=len(content))

    try:
        text = await asyncio.to_thread(load_text, stored_path)
        is_image = extension in {".png", ".jpg", ".jpeg", ".webp"}
        pdf_images = None
        if extension == ".pdf" and not text:
            try:
                page_bytes, page_count = await asyncio.to_thread(render_scanned_pdf, stored_path)
                pdf_images = [(page, "image/png") for page in page_bytes]
                emit("ocr.pdf_rendered", step=1, user_id=user_id, document_id=document_id, page_count=page_count)
            except ValueError as exc:
                raise HTTPException(413, str(exc)) from exc
        if not text and not is_image and pdf_images is None:
            raise HTTPException(422, "No readable text was found in the uploaded document")
        emit("upload.loaded", step=1, user_id=user_id, document_id=document_id, characters=len(text), requires_vision=is_image or pdf_images is not None)

        emit("extraction.started", step=2, user_id=user_id, document_id=document_id, model="gemini-3.8-flash")
        report = await asyncio.to_thread(extract_report, text, image_path=stored_path if is_image else None, image_data=pdf_images)
        if pdf_images is not None:
            report.extraction_notes.append("Scanned PDF pages were interpreted with vision OCR; verify all extracted details against the source.")
        uncertain_medications = [item.drug_name for item in report.medications if item.confidence < 0.85]
        if uncertain_medications:
            report.extraction_notes.append("Confirm low-confidence medication names or details with the source document and clinician.")
        report_data = report.model_dump(mode="json")
        emit("extraction.complete", step=2, user_id=user_id, document_id=document_id, report_type=report.report_type,
             diagnosis_count=len(report.diagnoses), lab_count=len(report.lab_results), medication_count=len(report.medications))

        searchable = f"Source document text:\n{text}\n\nStructured extraction:\n{serialize_report(report)}".strip()
        pieces = chunk_report(searchable, user_id=user_id, document_id=document_id, source_name=source_name,
                              report_date=str(report.report_date) if report.report_date else None)
        chunk_rows = [{"chunk_index": i, "text": item.page_content} for i, item in enumerate(pieces)]
        emit("chunking.complete", step=3, user_id=user_id, document_id=document_id, chunk_count=len(chunk_rows), strategy="recursive_character_text_splitter")

        database.save_document(document_id=document_id, user_id=user_id, source_name=source_name,
                               mime_type=file.content_type, storage_path=str(stored_path), report=report_data)
        database.add_report_memories(user_id=user_id, document_id=document_id, report=report_data)
        await asyncio.to_thread(vector_store.add, user_id=user_id, document_id=document_id, source_name=source_name, chunks=chunk_rows)
        emit("upload.complete", step=5, user_id=user_id, document_id=document_id, indexed_chunks=len(chunk_rows))
        return {"document_id": document_id, "source_name": source_name, "report": report_data, "chunk_count": len(chunk_rows)}
    except HTTPException:
        raise
    except Exception as exc:
        emit("upload.failed", user_id=user_id, document_id=document_id, error_type=type(exc).__name__)
        raise HTTPException(502, f"Document processing failed ({type(exc).__name__}); the upload was not indexed") from exc


@app.post("/conversations", status_code=status.HTTP_201_CREATED)
def new_conversation(user: dict = Depends(current_user)) -> dict:
    conversation_id = str(uuid4())
    database.create_conversation(conversation_id=conversation_id, user_id=user["id"])
    emit("conversation.created", user_id=user["id"], conversation_id=conversation_id)
    return {"conversation_id": conversation_id}


@app.post("/chat")
async def chat(payload: ChatRequest, user: dict = Depends(current_user)) -> dict:
    user_id = user["id"]
    conversation_id = payload.conversation_id or str(uuid4())
    if not payload.conversation_id:
        database.create_conversation(conversation_id=conversation_id, user_id=user_id)
        emit("conversation.created", user_id=user_id, conversation_id=conversation_id)
    try:
        database.save_message(conversation_id=conversation_id, user_id=user_id, role="user", content=payload.question)
    except ValueError as exc:
        raise HTTPException(404, "Conversation not found for this account") from exc
    emit("query.received", step=1, user_id=user_id, conversation_id=conversation_id, question_characters=len(payload.question))
    graph = build_query_graph(database, vector_store)
    compiled_view = graph.get_graph()
    emit(
        "graph.execution_plan",
        step=1,
        user_id=user_id,
        format="mermaid",
        diagram=compiled_view.draw_mermaid(),
    )
    print(compiled_view.draw_ascii(), flush=True)
    result = await graph.ainvoke({"user_id": user_id, "conversation_id": conversation_id, "question": payload.question})
    answer = result["answer"]
    database.save_message(conversation_id=conversation_id, user_id=user_id, role="assistant", content=answer)
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "retrieved_chunks": [
            {"document_id": item["document_id"], "source_name": item["metadata"]["source_name"], "chunk_index": item["chunk_index"], "similarity": item["similarity"]}
            for item in result.get("retrieved", [])
        ],
    }


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str, user: dict = Depends(current_user)) -> list[dict]:
    history = database.recent_messages(conversation_id=conversation_id, user_id=user["id"], limit=100)
    if not history:
        raise HTTPException(404, "Conversation not found or has no messages")
    return history
