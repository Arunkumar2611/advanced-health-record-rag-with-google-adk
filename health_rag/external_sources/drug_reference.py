import httpx

from health_rag.config import OPENFDA_BASE_URL, RXNORM_BASE_URL
from health_rag.observability import emit


async def lookup_drug(drug_name: str) -> dict | None:
    """Look up identity and label details; never returns personalized dosing advice."""
    normalized = drug_name.strip()
    if not normalized:
        return None
    emit("drug_lookup.started", sources=["RxNorm", "OpenFDA"])
    result: dict = {}
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        try:
            rx = await client.get(f"{RXNORM_BASE_URL}/rxcui.json", params={"name": normalized})
            rx.raise_for_status()
            group = rx.json().get("idGroup", {})
            rxcuis = group.get("rxnormId", [])
            if rxcuis:
                props_response = await client.get(f"{RXNORM_BASE_URL}/rxcui/{rxcuis[0]}/properties.json")
                props_response.raise_for_status()
                props = props_response.json().get("properties", {})
                result["rxnorm"] = {"source": "NIH RxNorm", "rxcui": rxcuis[0], "name": props.get("name"), "tty": props.get("tty")}
        except (httpx.HTTPError, ValueError) as exc:
            emit("drug_lookup.source_error", source="RxNorm", reason=type(exc).__name__)
        try:
            safe_name = normalized.replace('"', "")
            fda = await client.get(f"{OPENFDA_BASE_URL}/drug/label.json", params={"search": f'openfda.brand_name:"{safe_name}" OR openfda.generic_name:"{safe_name}"', "limit": 1})
            if fda.status_code == 200:
                item = fda.json().get("results", [{}])[0]
                openfda = item.get("openfda", {})
                result["fda_label"] = {
                    "source": "US FDA drug label",
                    "generic_name": openfda.get("generic_name", [None])[0],
                    "brand_name": openfda.get("brand_name", [None])[0],
                    "indications": item.get("indications_and_usage", [])[:2],
                    "warnings": item.get("warnings", [])[:2],
                    "source_note": "Regulatory labeling; not individualized medical advice.",
                }
        except (httpx.HTTPError, ValueError) as exc:
            emit("drug_lookup.source_error", source="OpenFDA", reason=type(exc).__name__)
        if result:
            emit("drug_lookup.complete", found=True, source_names=list(result))
            return result
    emit("drug_lookup.complete", found=False)
    return None
