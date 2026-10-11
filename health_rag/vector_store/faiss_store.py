import json
import threading
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from health_rag.config import EMBEDDING_MODEL, FAISS_DIR
from health_rag.observability import emit
from health_rag.persistence.database import Database


class UserFaissStore:
    """Separate normalized inner-product FAISS index per authenticated user."""

    def __init__(self, database: Database, root: Path = FAISS_DIR) -> None:
        self.database = database
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self._indexes: dict[str, faiss.Index] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def _lock(self, user_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(user_id, threading.RLock())

    def _path(self, user_id: str) -> Path:
        return self.root / f"{user_id}.faiss"

    def _load(self, user_id: str, dimension: int | None = None) -> faiss.Index:
        with self._lock(user_id):
            if user_id not in self._indexes:
                path = self._path(user_id)
                if path.exists():
                    self._indexes[user_id] = faiss.read_index(str(path))
                else:
                    if dimension is None:
                        dimension = len(self.embeddings.embed_query("dimension probe"))
                    self._indexes[user_id] = faiss.IndexFlatIP(dimension)
            return self._indexes[user_id]

    def add(self, *, user_id: str, document_id: str, source_name: str, chunks: list[dict]) -> None:
        if not chunks:
            return
        emit("embedding.started", step=4, user_id=user_id, document_id=document_id, chunks=len(chunks))
        vectors = np.asarray(self.embeddings.embed_documents([chunk["text"] for chunk in chunks]), dtype="float32")
        faiss.normalize_L2(vectors)
        with self._lock(user_id):
            index = self._load(user_id, dimension=vectors.shape[1])
            first = index.ntotal
            index.add(vectors)
            mapped = []
            for offset, chunk in enumerate(chunks):
                mapped.append({
                    "id": uuid4().hex,
                    "vector_index": first + offset,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "metadata": {"user_id": user_id, "document_id": document_id, "source_name": source_name},
                })
            self.database.add_chunks(user_id=user_id, document_id=document_id, chunks=mapped)
            temporary = self._path(user_id).with_suffix(".tmp")
            faiss.write_index(index, str(temporary))
            temporary.replace(self._path(user_id))
            emit("embedding.complete", step=4, user_id=user_id, vectors_added=len(chunks), total_vectors=index.ntotal)

    def search(self, *, user_id: str, query: str, limit: int = 6) -> list[dict]:
        if user_id not in self._indexes and not self._path(user_id).exists():
            return []
        index = self._load(user_id)
        if index.ntotal == 0:
            return []
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, indices = index.search(query_vector, min(limit, index.ntotal))
        results = []
        with self.database.connect() as db:
            for score, vector_index in zip(scores[0].tolist(), indices[0].tolist()):
                if vector_index < 0:
                    continue
                row = db.execute("SELECT document_id,chunk_index,text,metadata_json FROM chunks WHERE user_id=? AND vector_index=?", (user_id, vector_index)).fetchone()
                if row:
                    results.append({"document_id": row["document_id"], "chunk_index": row["chunk_index"], "text": row["text"], "metadata": json.loads(row["metadata_json"]), "similarity": round(float(score), 4)})
        return results
