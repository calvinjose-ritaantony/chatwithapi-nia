from pydantic import BaseModel, Field

class ModelConfiguration(BaseModel):
    max_tokens: int = Field(default=300, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=1, description="Nucleus sampling probability")
    frequency_penalty: float = Field(default=0, description="Frequency penalty")
    presence_penalty: float = Field(default=0, description="Presence penalty")
    web_search: bool = Field(default=False, description="Enable web search for additional context")