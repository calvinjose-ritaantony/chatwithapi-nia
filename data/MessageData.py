from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId

class Message(BaseModel):
    _id: ObjectId
    gpt_id: str = Field(..., description="Identifier for the associated GPT model")
    role: str = Field(..., description="Role of the message sender (e.g., user, assistant)")
    content: str = Field(..., description="Content of the message")
    use_case_id: str = Field(..., description="Use case identifier")
    user_name: str = Field(..., description="Name of the user who sent the message")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    hiddenFlag: bool = Field(default=False, description="Flag to indicate if the message is hidden")