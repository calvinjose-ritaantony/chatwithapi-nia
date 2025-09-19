from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId

class GPTData(BaseModel):
    _id: ObjectId
    name: str = Field(description="Name of the GPT model")
    description: str = Field(..., description="Description of the GPT model")
    instructions: str = Field(..., description="Instructions or System prompt for the GPT model")
    use_rag: bool = Field(default=False, description="Whether to use Retrieval-Augmented Generation")
    user: str = Field(..., description="User associated with the GPT model")
    use_case_id: str = Field(..., description="Use case identifier")
    #token_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")

# Sample usage for field
#name: str = Field(..., min_length=1, max_length=100, description="Name must be between 1 and 100 characters")
#age: int = Field(..., gt=0, le=100, description="Age must be a positive integer")

# Error 
#The error occurs because ... is actually Python's Ellipsis object, which Pydantic uses as a sentinel value for required fields. When you use both ... and default=, it's trying to set two default values.
