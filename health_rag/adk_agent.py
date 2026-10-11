"""Google ADK agent used for grounded response generation."""
from google.adk.agents import Agent
from google.adk.models import Gemini

from health_rag.config import DISCLAIMER, MODEL_NAME

root_agent = Agent(
    name="personal_health_rag_agent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "Summarize diagnoses as documented in supplied health records; do not independently diagnose. "
        "Use supplied report excerpts and explicitly labeled external medicine references. Never recommend treatment or invent doses. If evidence is absent or weak, say so. "
        f"Append this exact disclaimer to every answer: {DISCLAIMER}"
    ),
)
