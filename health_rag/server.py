import os

import uvicorn

from health_rag.config import JWT_SECRET


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    unsafe_secret = JWT_SECRET == "dev-only-change-this-secret-before-deployment" or JWT_SECRET.startswith("replace_with_") or len(JWT_SECRET) < 32
    if unsafe_secret and (os.getenv("ENVIRONMENT") == "production" or host not in {"127.0.0.1", "localhost"}):
        raise SystemExit("Set HEALTH_RAG_JWT_SECRET to a random value of at least 32 characters before exposing the server")
    uvicorn.run("health_rag.api.app:app", host=host, port=int(os.getenv("PORT", "8000")), reload=False)


if __name__ == "__main__":
    main()
