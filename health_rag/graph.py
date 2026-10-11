import asyncio
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from health_rag.answering import generate_answer
from health_rag.config import DISCLAIMER, MAX_CONTEXT_CHUNKS, MIN_RELEVANCE_SCORE, SHORT_TERM_TURNS
from health_rag.external_sources.drug_reference import lookup_drug
from health_rag.observability import emit
from health_rag.persistence.database import Database
from health_rag.vector_store.faiss_store import UserFaissStore


class QueryState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    question: str
    retrieved: list[dict]
    drug_reference: dict | None
    answer: str


RED_FLAGS = (
    r"severe (?:rectal )?bleeding", r"vomiting blood", r"severe abdominal pain",
    r"difficulty breathing", r"can't breathe", r"cannot breathe", r"suicid(?:e|al)",
)
MEDICATION_REQUESTS = (
    r"suggest .*medicines?", r"recommend .*medicines?", r"what medicine should i (?:take|use)",
    r"what medication should i", r"medicine for (?:my|the) (?:cancer|blood cancer|symptoms|condition)",
    r"which drugs? should i (?:take|use)", r"prescribe .*", r"what treatment should i (?:take|get|start)",
    r"what drugs? can i take", r"what medicines? can i take", r"medication for my cancer",
)
DRUG_QUESTION = re.compile(r"(?:why is|what is|tell me about|uses of|what does)\s+([A-Za-z][A-Za-z0-9-]{2,})", re.I)
GENERIC_DRUG_REFERENCES = {"this", "that", "my", "the", "medicine", "medication", "drug", "prescription", "it", "useful"}


def build_query_graph(database: Database, vector_store: UserFaissStore):
    async def safety_router(state: QueryState) -> QueryState:
        question = state["question"].lower()
        if any(re.search(pattern, question) for pattern in RED_FLAGS):
            emit("safety.red_flag", step=1, user_id=state["user_id"])
            return {"answer": "Your message includes a potentially urgent symptom. Please seek immediate medical care or contact your local emergency service now. " + DISCLAIMER, "retrieved": []}
        if any(re.search(pattern, question) for pattern in MEDICATION_REQUESTS):
            emit("safety.medication_recommendation_blocked", step=1, user_id=state["user_id"])
            return {"answer": "I can’t recommend or prescribe medicines. Treatment for a condition such as blood cancer must be chosen by the treating specialist. I can explain medicines documented on your uploaded prescription or explain general information about a named medicine from a drug reference, but I can’t tell you to start or change treatment. " + DISCLAIMER, "retrieved": []}
        return {}

    async def retrieve(state: QueryState) -> QueryState:
        emit("retrieval.started", step=2, user_id=state["user_id"], query_characters=len(state["question"]))
        found = await asyncio.to_thread(vector_store.search, user_id=state["user_id"], query=state["question"], limit=MAX_CONTEXT_CHUNKS)
        relevant = [item for item in found if item["similarity"] >= MIN_RELEVANCE_SCORE]
        for item in found:
            emit("retrieval.match", user_id=state["user_id"], document_id=item["document_id"], chunk_index=item["chunk_index"], similarity=item["similarity"], included=item in relevant)
        if not relevant:
            emit("retrieval.no_relevant_evidence", user_id=state["user_id"])
        return {"retrieved": relevant}

    async def lookup_external_drug(state: QueryState) -> QueryState:
        question = state["question"].lower()
        medicines = database.find_medications(state["user_id"])
        medicine_name = next((item.get("drug_name") for item in medicines if item.get("drug_name") and item["drug_name"].lower() in question), None)
        if medicine_name is None and len(medicines) == 1 and re.search(r"\bmy (?:medicine|medication|drug)\b", question):
            medicine_name = medicines[0].get("drug_name")
        explicit = DRUG_QUESTION.search(state["question"])
        if medicine_name is None and explicit and explicit.group(1).lower() not in GENERIC_DRUG_REFERENCES:
            medicine_name = explicit.group(1)
        if not medicine_name:
            return {"drug_reference": None}
        # Reuse document evidence if it already explains the purpose; otherwise try reference sources.
        local_explanation = any(
            medicine_name.lower() in chunk["text"].lower()
            and re.search(r"\b(used for|indication|purpose|treats|prescribed for)\b", chunk["text"], re.I)
            for chunk in state.get("retrieved", [])
        )
        if local_explanation:
            emit("drug_lookup.local_evidence_sufficient", step=3, user_id=state["user_id"])
            return {"drug_reference": None}
        # A local prescription record is the patient's evidence; public references only explain general use.
        local_match = next((item for item in medicines if item.get("drug_name", "").lower() == medicine_name.lower()), None)
        if local_match and any(term in question for term in ("why", "what is", "useful", "purpose", "for")):
            emit("drug_lookup.local_prescription_match", step=3, user_id=state["user_id"], document_id=local_match.get("source_document_id"))
        reference = await lookup_drug(medicine_name)
        return {"drug_reference": reference}

    async def respond(state: QueryState) -> QueryState:
        profile = database.user_profile_context(state["user_id"])
        history = database.recent_messages(conversation_id=state["conversation_id"], user_id=state["user_id"], limit=SHORT_TERM_TURNS * 2)
        answer = await generate_answer(
            state["question"], state.get("retrieved", []), patient_profile=profile,
            chat_history=history[:-1], drug_reference=state.get("drug_reference"),
        )
        emit("response.complete", step=4, user_id=state["user_id"], retrieved_chunks=len(state.get("retrieved", [])), answer_characters=len(answer))
        return {"answer": answer}

    builder = StateGraph(QueryState)
    builder.add_node("safety_router", safety_router)
    builder.add_node("retrieve", retrieve)
    builder.add_node("lookup_external_drug", lookup_external_drug)
    builder.add_node("respond", respond)
    builder.add_edge(START, "safety_router")
    builder.add_conditional_edges("safety_router", lambda state: "stop" if state.get("answer") else "retrieve", {"stop": END, "retrieve": "retrieve"})
    builder.add_edge("retrieve", "lookup_external_drug")
    builder.add_edge("lookup_external_drug", "respond")
    builder.add_edge("respond", END)
    return builder.compile()
