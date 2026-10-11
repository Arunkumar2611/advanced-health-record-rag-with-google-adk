from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=160)
    password: str = Field(min_length=12, max_length=256)


class TokenRequest(BaseModel):
    username: EmailStr
    password: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    conversation_id: str | None = None


class ConversationResponse(BaseModel):
    conversation_id: str
