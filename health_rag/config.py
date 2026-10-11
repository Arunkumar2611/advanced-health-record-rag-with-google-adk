from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DATA_DIR = Path(os.getenv("HEALTH_RAG_DATA_DIR", "./data")).resolve()
DATABASE_PATH = Path(os.getenv("HEALTH_RAG_DATABASE", str(DATA_DIR / "health_rag.sqlite3"))).resolve()
FAISS_DIR = DATA_DIR / "faiss"
JWT_SECRET = os.getenv("HEALTH_RAG_JWT_SECRET", "")
JWT_ISSUER = os.getenv("HEALTH_RAG_JWT_ISSUER", "personal-health-rag")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
RXNORM_BASE_URL = os.getenv("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST").rstrip("/")
OPENFDA_BASE_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov").rstrip("/")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
MIN_RELEVANCE_SCORE = 0.25
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_CONTEXT_CHUNKS = 6
SHORT_TERM_TURNS = 8
DISCLAIMER = (
    "This is a personal health information assistant, not a medical diagnosis or "
    "prescription. Confirm clinical decisions with your licensed care team."
)

if not JWT_SECRET:
    # Development only: deployment must configure a secret via environment.
    JWT_SECRET = "dev-only-change-this-secret-before-deployment"
