import json

from pydantic import BaseModel

class ModelResponse(BaseModel):
    model_response: str
    total_tokens: int
    follow_up_questions: list[str]
    reasoning: str
    is_table: bool
    is_graph: bool
    is_code: bool
    error_message: str = ""
   
    
    