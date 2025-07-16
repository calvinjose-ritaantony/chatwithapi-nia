from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class HandleFAQsOutput(BaseModel):
    faq_question: str
    faq_answer: str
    faq_topic: str
    support_contact: str
