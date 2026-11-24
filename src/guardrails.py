from pydantic import BaseModel, Field, field_validator
import re

MAX_LEN = 2000
SUSPICIOUS = re.compile(r"(ignore previous|__import__|\.\./|<script|;drop\s)", re.I)

class MessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_LEN, description="Check karne ke liye SMS ya Email")

    @field_validator("message")
    @classmethod
    def check_malicious_input(cls, v: str):
        if not v.strip():
            raise ValueError("Message can not be empty")
        if SUSPICIOUS.search(v):
            raise ValueError("Security Alert: Suspicious pattern detected in input!")
        return v.strip()
