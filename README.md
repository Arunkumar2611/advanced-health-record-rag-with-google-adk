# Personal Health RAG

An account-scoped health-record RAG application built with Google ADK, LangChain, LangGraph, Gemini `gemini-3.8-flash`, and FAISS. A user signs up and signs in; the server assigns a UUID and derives the account scope from the bearer token. Upload and chat requests never accept a user ID from the browser.

## Structure

```text
health_rag/
  api/                  FastAPI routes and request schemas
  chunking/             Medical document chunking
  data_loading/         PDF, DOCX, text and image input
  external_sources/     RxNorm and FDA medicine references
  persistence/          SQLite accounts, reports, facts and conversations
  security/             Password hashing and token authentication
  static/               Browser sign-in, upload and chat UI
  vector_store/         Per-user normalized Gemini embeddings in FAISS
  answering.py          ADK grounded answer generation
  extraction.py         LangChain/Gemini typed report extraction
  graph.py              LangGraph safety, retrieval and answer flow
  observability.py       Structured print-only event output
  server.py              Local API server entry point
```

## Run locally

Requires Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` and a long random `HEALTH_RAG_JWT_SECRET` in `.env`, then run:

```powershell
health-rag
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Register an account, sign in, upload appointment cards, prescriptions, and reports, then ask questions in that account's conversation. The backend assigns the user ID after registration.

## Ingestion and retrieval

The upload path validates format and size, extracts text or sends images for Gemini vision extraction, validates a Pydantic report schema, stores the source and structured report, creates sentence/paragraph-aware chunks, embeds them with Gemini, and adds normalized vectors to that user's FAISS inner-product index. FAISS similarity scores are cosine similarity in `[0, 1]`; the API returns the scores and source chunks used. The score measures text relevance, not medical certainty.

SQLite stores account records, report metadata and extracted JSON, clinically structured long-term facts, and conversation messages. Conversation turns provide persistent short-term context; reports and extracted diagnoses/labs/medications form the long-term patient context. A separate FAISS index per account plus the token-derived user scope prevents cross-account retrieval.

Medicine questions first check the patient's report chunks and medication records. If the local records do not explain a named medicine, the workflow looks up general drug identity/label details in NIH RxNorm and the US FDA label API. Reference information is presented as general information, never personalized treatment advice.

## Safety behavior

- Requests to recommend or prescribe medication are blocked. The agent may summarize a prescription it can cite, but it cannot recommend starting, stopping, or changing treatment.
- An uploaded diagnosis or disease duration is described only when the source document explicitly states it. Lab values retain the recorded unit and range; isolated values are not converted into a diagnosis.
- A small explicit red-flag phrase list returns urgent-care guidance before model generation. This is not a complete emergency triage system.
- Report extraction is fallible. Review every extracted diagnosis, date, dose, and lab result against the source document. Model-generated confidence is not calibrated.
- Supported formats are text, Markdown, PDF, DOCX, and PNG/JPEG/WEBP images. Text-based PDFs use native text extraction; scanned PDFs are rendered page-by-page for Gemini vision OCR, up to 25 pages per upload.

## Structured terminal progress

All application progress is emitted as one JSON object per line through `print`; no logging library is used. Before each chat graph runs, the terminal prints its compiled ASCII diagram and emits the Mermaid representation as a `graph.execution_plan` JSON event. Events have UTC timestamps, stable event names, and step numbers, for example `upload.received` (1), `extraction.complete` (2), `chunking.complete` (3), `embedding.complete` (4), and `upload.complete` (5). Passwords, tokens, report contents, and question text are not written to progress events.

## Deployment boundary

This is a production-oriented application foundation, not a compliance-certified clinical deployment. Before handling real patient data or exposing it publicly, deploy behind HTTPS, set strong secret management, encrypt database/uploads/vector indexes at rest, add rate limiting and account recovery/MFA, apply schema migrations and backup/retention policies, expand OCR coverage and human confirmation for uncertain extraction, and perform privacy, security, clinical safety, and load reviews. Local SQLite and single-process FAISS are intended for a single-node deployment; use a managed relational database and tenant-aware vector service for horizontal scaling.
