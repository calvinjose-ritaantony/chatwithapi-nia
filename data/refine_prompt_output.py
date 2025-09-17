from pydantic import BaseModel, Field
from typing import Literal

class EvaluationItem(BaseModel):
    """Represents an evaluation criterion with adherence status and justification."""

    adherence: Literal["true", "false"] = Field(..., description="Indicates if the prompt adheres to the specific criterion.")
    justification: str = Field(..., description="An explanation supporting the adherence status.")

class EvaluationDefinitions(BaseModel):
    """Holds various evaluation criteria for prompt assessment."""

    task_definition: EvaluationItem = Field(..., description="Clarity and specificity of the task definition.")
    output_format: EvaluationItem = Field(..., description="Appropriateness and clarity of the expected output format.")
    scope_and_constraints: EvaluationItem = Field(..., description="Defined scope and constraints for the task.")
    input_data: EvaluationItem = Field(..., description="Relevance and clarity of the input data provided.")
    clarity_check: EvaluationItem = Field(..., description="Overall clarity of the prompt.")
    example_inclusion: EvaluationItem = Field(..., description="Inclusion of relevant examples to guide the response.")
    edge_case_handling: EvaluationItem = Field(..., description="Consideration of potential edge cases.")
    tone_and_persona: EvaluationItem = Field(..., description="Specified tone and persona for the response.")
    ethical_guardrails: EvaluationItem = Field(..., description="Incorporation of ethical considerations and guardrails.")
    performance_optimization: EvaluationItem = Field(..., description="Guidelines for optimizing performance and efficiency.")
    testability: EvaluationItem = Field(..., description="Ease of testing and validating the prompt's effectiveness.")
    domain_relevance: EvaluationItem = Field(..., description="Relevance to the specific domain or industry.")

class RefinePromptOutput(BaseModel):
    """Schema for the output of the prompt refinement process."""

    title: str = Field(..., description="A concise title for the prompt.")
    pre_evaluation: EvaluationDefinitions = Field(..., description="Evaluation of the original prompt based on defined criteria.")
    complexity_assessment: Literal["simple", "complex"] = Field(..., description="Assessment of the prompt's complexity.")
    needs_refinement: bool = Field(..., description="Indicates whether the prompt requires refinement.")
    refinement_reason: str = Field(..., description="Justification for the need for refinement.")
    refined_prompt: str = Field(..., description="The improved version of the original prompt.")
    post_evaluation: EvaluationDefinitions = Field(..., description="Evaluation of the refined prompt based on defined criteria.")