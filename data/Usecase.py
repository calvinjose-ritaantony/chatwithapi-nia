from pydantic import BaseModel
from datetime import datetime
from bson import ObjectId
from typing import List, Dict
from data.ModelConfiguration import ModelConfiguration

class Prompt(BaseModel):
    role: str
    prompt: str
    key: str
    title: str
    user: str



class Usecase(BaseModel): 
    _id: ObjectId
    gpt_id: str
    name: str
    description: str
    instructions: str
    index_name: str
    semantic_configuration_name: str
    prompts: List[Prompt]
    created_at: str = datetime.now().isoformat()
    user_message: str
    fields_to_select: List[str]
    document_count: int
    role_information: str
    model_configuration: ModelConfiguration
