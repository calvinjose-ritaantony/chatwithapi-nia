from pydantic import BaseModel
from typing import List, Literal, Optional, Union
from datetime import datetime

class EvaluationItem(BaseModel):
    adherence: Literal["true", "false"]
    justification: str

class EvaluationDefinitions(BaseModel):
    task_definition: EvaluationItem
    output_format: EvaluationItem
    scope_and_constraints: EvaluationItem
    input_data: EvaluationItem
    clarity_check: EvaluationItem
    example_inclusion: EvaluationItem
    edge_case_handling: EvaluationItem
    tone_and_persona: EvaluationItem
    ethical_guardrails: EvaluationItem
    performance_optimization: EvaluationItem
    testability: EvaluationItem
    domain_relevance: EvaluationItem

class RefinePromptOutput(BaseModel):
    title: str
    pre_evaluation: EvaluationDefinitions
    complexity_assessment: Literal["simple", "complex"]
    needs_refinement: bool
    refinement_reason: str
    refined_prompt: str
    post_evaluation: EvaluationDefinitions