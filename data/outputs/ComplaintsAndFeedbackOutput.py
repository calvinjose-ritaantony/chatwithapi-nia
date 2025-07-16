from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ComplaintsAndFeedbackOutput(BaseModel):
    product_id: str
    complaint_id: str
    feedback: str
    sentiment: str
    action_taken: str
    resolved_date: Optional[datetime] = None
    escalation_level: str
