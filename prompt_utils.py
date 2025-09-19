import json
import logging
import re
import os

from openai import AsyncAzureOpenAI
from openai.types.chat import ParsedChatCompletion
from PromptValidationResult import PromptValidationResult
from typing import Optional, Dict
from azure_openai_utils import (GPT_4o_ENDPOINT_URL, GPT_4o_API_KEY, GPT_4o_API_VERSION, GPT_4o_MODEL_NAME)
from data.refine_prompt_output import EvaluationDefinitions, RefinePromptOutput

logger = logging.getLogger(__name__)

class PromptValidator:
    """Class to validate and refine prompts using an LLM for analysis."""
    
    # Define minimum and maximum parameters
    MINIMUM_PARAMETERS = [
        "task_definition",
        "output_format",
        "scope_and_constraints",
        "input_data",
        "clarity_check"
    ]

    MAXIMUM_PARAMETERS = [
        "example_inclusion",
        "edge_case_handling",
        "tone_and_persona",
        "ethical_guardrails",
        "performance_optimization",
        "testability",
        "domain_relevance"
    ]
    
    ALL_PARAMETERS = MINIMUM_PARAMETERS + MAXIMUM_PARAMETERS
    
    def __init__(self, llm_api=None):
        """Initialize with optional LLM API client (e.g., xAI API for Grok 3)."""
        self.llm_api = llm_api  # Placeholder for LLM API client
    
    async def call_llm(self, instruction: str, temperature: float = 0.3, top_p: float = 0.95) -> str:
        """
        Call the LLM API with enhanced prompt engineering capabilities.
        """
        
        # Default expert prompt engineer persona if no custom system prompt is provided
        prompt_refining_system_prompt = """You are an elite prompt engineering expert with extensive experience in analyzing, evaluating, and optimizing prompts.
            Your capabilities include:
            1. Rapid prompt dissection and structured analysis against established parameters
            2. Evidence-based evaluation with clear reasoning for each assessment
            3. Strategic prompt refinement that transforms weaknesses into strengths
            4. Comparative assessment of original vs. refined prompts with quantifiable improvements
            5. Pattern recognition across diverse prompt types and use cases

            Approach each prompt methodically:
            - Analyze without assumptions
            - Provide specific, actionable feedback with reasoning
            - Optimize for clarity, specificity, and effective constraint implementation
            - Ensure all output follows requested JSON schema precisely
            - Balance technical rigor with practical usability"""

        conversations = [
            {"role": "system", "content": prompt_refining_system_prompt},
            {"role": "user", "content": instruction}
        ]

        azure_openai_client = AsyncAzureOpenAI(
            azure_endpoint=GPT_4o_ENDPOINT_URL, 
            api_key=GPT_4o_API_KEY, 
            api_version=GPT_4o_API_VERSION)
        
        # Retry logic for model call (similar to your reference)
        response = None
        retry_models = [GPT_4o_MODEL_NAME]
        i = 1
        while True:
            model_env_var = f"GPT_RETRY_MODELS_{i}"
            alt_model = os.getenv(model_env_var)
            if not alt_model:
                break
            retry_models.append(alt_model)
            i += 1

        last_exception = None
        for model_name in retry_models:
            try:
                response: ParsedChatCompletion  = await azure_openai_client.chat.completions.parse(
                model=model_name,
                messages=conversations,
                max_tokens=1200,  # Increased to accommodate justifications
                temperature=temperature,
                top_p=top_p,
                response_format=RefinePromptOutput
            )
                # If response is valid, break out of retry loop
                if response and response.choices and response.choices[0].message.content:
                    break
            except Exception as ex:
                logger.warning(f"Model '{model_name}' failed: {ex!s}", exc_info=True)
                last_exception = ex
            continue

        if response is None or not response.choices or not response.choices[0].message.content:
            logger.error("All models failed to return a valid response.")
            raise last_exception if last_exception else ValueError("No valid response from any model.")

        
        model_response = response.choices[0].message.content
        return model_response
    
    async def process_prompt_optimized(self, prompt: str, system_prompt: Optional[str] = None) -> PromptValidationResult:
        """
        Enhanced prompt processing that analyzes, evaluates with justification, refines, and re-evaluates prompts.
        
        Args:
            prompt: The user prompt to be evaluated and refined
            system_prompt: Optional system prompt that provides context or guidance
            
        Returns:
            PromptValidationResult object containing the refined prompt, parameter adherence, justifications,
            complexity assessment, and refinement reasoning
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Create the evaluation and refinement instruction
        refinement_instruction = f"""
        As an expert prompt engineer, analyze the given prompt and provide a detailed evaluation and enhanced refinement of the given prompt. Follow the steps diligently to generate the final response.

        - Step 1: UNDERSTANDING THE CONTEXT AND INTENT
        Carefully read the provided system and user prompts to fully grasp the context, objectives, and intended outcomes.
        Identify any implicit goals or requirements that may not be explicitly stated.
        
        - Step 2 : ANALYSIS
        Analyze the prompt against the parameters provided in EvaluationDefinitions class. 
        
        - Step 3:  EVALUATION
        Evaluate the prompt against the parameters in EvaluationDefinitions class and generate EvaluationItem class with detailed justification, adherence for each parameter.
        
        # Step 4: REFINEMENT
        Determine prompt complexity (simple or complex), whether it needs refinement, and if so, provide detailed recommendations.
        
        # PART 4: PROMPT ENGINEERING
        Create a refined version of the prompt that addresses all identified issues, optimizes parameter adherence, and maintains the original intent. Ensure the refined prompt is clear, specific, and actionable.  
              
        # PART 5: COMPARATIVE EVALUATION
        Re-evaluate the refined prompt against the same parameters to demonstrate improvements.
        
        # INPUTS
        System prompt: {system_prompt or 'None'}
        User prompt: {prompt}

        """
        
        # Call the LLM with the full analysis instruction
        try:
            analysis_response = await self.call_llm(refinement_instruction, temperature=0.2, top_p=0.95)
            logger.info(f"LLM Analysis Response: {analysis_response}")
            # analysis_json = self.extract_json_from_response(analysis_response)
            data = json.loads(analysis_response)
            logger.info(f"Parsed Analysis Data: {data}")
            
            # Extract adherence values for pre-evaluation
            pre_evaluation_result = data.get("pre_evaluation", {})
            post_evaluation_result = data.get("post_evaluation", {})
            title_result = data.get("title", "Simple Prompt")
            
            # Extract refined prompt (use original if refinement was not needed)
            needs_refinement = data.get("needs_refinement", "false")
            refined_prompt = data.get("refined_prompt", prompt) if needs_refinement else prompt
            
            # Create and return the validation result with justifications
            return PromptValidationResult(
                title=title_result,
                original_prompt=prompt,
                refined_prompt=refined_prompt,
                pre_evaluation_result=pre_evaluation_result,
                post_evaluation_result=post_evaluation_result,
                complexity_assessment=data.get("complexity_assessment", "simple"),
                refinement_reason=data.get("refinement_reason", "No refinement needed")
            )
            
        except Exception as e:
            logger.error(f"Error during prompt analysis and refinement: {str(e)}")
            # Fallback to return original prompt in case of errors
            return PromptValidationResult(prompt, {"error": f"Analysis failed: {str(e)}"})