from pydantic import BaseModel, Field
from typing import List, Optional

    
class BasePromptTemplate(BaseModel):
    """Base class for AI prompt templates that can be extended for specific use cases"""

    role: str = Field(
        default="You are an AI assistant.",
        description="The role definition for the AI assistant"
    )

    context: str = Field(
        default="<query>{query}</query>",
        description="The user query wrapped in XML tags for clear identification"
    )

    task: List[str] = Field(
        default=[
            "Use the given context information and generate a proper response to the user query.",
        ],
        description="List of specific tasks the AI should perform"
    )

    flow: List[str] = Field(
        default=[],
        description="The workflow steps to follow when responding"
    )

    data_sources: dict = Field(
        default={
            "retrieved_data": "{sources}",
            "web_search_results": "{web_search_results}"
        },
        description="Available data sources for context"
    )

    reasoning_steps: List[str] = Field(
        default=[],
        description="Step-by-step reasoning process to follow"
    )

    response_format: List[str] = Field(
        default=[],
        description="How to structure the response to the user"
    )

    description: Optional[str] = Field(
        default=None,
        description="A brief description of the use case"
    )

    semantic_configuration_name: Optional[str] = Field(
        default=None,
        description="Name of the semantic configuration to use"
    )

    index_name: Optional[str] = Field(
        default=None,
        description="Name of the index to query for context"
    )

    fields_to_select: Optional[List[str]] = Field(
        default=[],
        description="Specific fields to select from the index"
    )

    document_count: Optional[int] = Field(
        default=None,
        description="Number of documents to retrieve"
    )

    model_configuration: Optional[dict] = Field(
        default=None,
        description="Configuration settings for the AI model"
    )

    prompts: Optional[List[dict]] = Field(
        default=[],
        description="List of prompts to use for the AI"
    )

    def format_prompt(self, query: str, sources: str, web_search_results: Optional[str] = "") -> str:
        """
        Format the prompt template with the provided query and sources.
        
        Args:
            query: The user's query
            sources: Retrieved data sources
            web_search_results: Optional web search results
            
        Returns:
            A formatted prompt string
        """
        formatted_context = self.context.replace("{query}", query)
        
        # Format the data sources
        data_context = f"RETRIEVED DATA: {sources}\n"
        if web_search_results:
            data_context += f"WEB SEARCH RESULTS: {web_search_results}"
        
        # Build the formatted prompt
        prompt_parts = [
            self.role,
            f"\nCONTEXT: The user query is: {formatted_context}"
        ]
        
        if self.task:
            prompt_parts.extend([
                "\nTASK:",
                "\n".join(f"- {task}" for task in self.task)
            ])
        
        if self.flow:
            prompt_parts.extend([
                "\nFLOW:",
                "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.flow))
            ])
        
        prompt_parts.append(f"\nContext:\n{data_context}")
        
        if self.reasoning_steps:
            prompt_parts.extend([
                "\nREASONING STEPS:",
                "\n".join(f"Step {i+1} - {step}" for i, step in enumerate(self.reasoning_steps))
            ])
        
        if self.response_format:
            prompt_parts.extend([
                "\nFORMAT:",
                "\n".join(f"- {fmt}" for fmt in self.response_format)
            ])
        
        return "\n".join(prompt_parts)

    def to_dict(self) -> dict:
        """
        Convert the prompt to a dictionary structure for JSON serialization
        
        Returns:
            Dictionary representation of the prompt
        """
        return {
            "role": self.role,
            "context": self.context,
            "task": self.task,
            "flow": self.flow,
            "data_sources": self.data_sources,
            "reasoning_steps": self.reasoning_steps,
            "response_format": self.response_format
        }