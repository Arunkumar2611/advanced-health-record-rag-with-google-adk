"""Small structured console events; intentionally uses print, never a logging library."""
import json
from datetime import datetime, timezone
from typing import Any


def emit(event: str, *, step: int | None = None, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if step is not None:
        payload["step"] = step
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)
