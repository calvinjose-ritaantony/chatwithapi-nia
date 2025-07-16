from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ManageTicketsOutput(BaseModel):
    incident_id: str
    title: str
    problem: str
    chat_log: str
