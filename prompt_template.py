import logging
from utils.BasePromptTemplate import BasePromptTemplate
import asyncio
import fastapi
import logging
from fastapi import APIRouter, HTTPException
from mongo_service import get_usecases, get_usecases_list


#fetch use case configurations from MongoDB for the specified gpt_id
async def build_prompt_templates(gpt_id: str):
    usecases_list = await get_usecases(gpt_id)
    USE_CASE_CONFIG = {uc["name"]: uc for uc in usecases_list if "name" in uc}

    prompt_templates = {}

    for name, config in USE_CASE_CONFIG.items():
        logging.info(f"use case '{name}' config: {config}")
        
        prompt_templates[name] = BasePromptTemplate(
            role=config.get("role_information", "You are an AI assistant."),
            context=config.get("user_message", "<query>{query}</query>"),
            task=[config.get("instructions", "")],
            flow=config.get("flow", []),
            data_sources={
                "retrieved_data": "{sources}",
                "web_search_results": "{web_search_results}"
            },
            reasoning_steps=config.get("reasoning_steps", []),
            response_format=config.get("response_format", []),

            description=config.get("description"),
            semantic_configuration_name=config.get("semantic_configuration_name"),
            index_name=config.get("index_name"),
            fields_to_select=config.get("fields_to_select", []),
            document_count=config.get("document_count"),
            model_configuration=config.get("model_configuration"),
            prompts=config.get("prompts", [])
        )

    print("********Prompt templates dict :", prompt_templates)
    return prompt_templates


router = APIRouter()


@router.get("/{gpt_id}")
async def get_prompt_templates(gpt_id: str):
    
    try:
        templates = await build_prompt_templates(gpt_id)
        # Convert BasePromptTemplate objects to dicts for JSON response
        response = {
            name: template.__dict__ for name, template in templates.items()
        }
        return response
    except Exception as e:
        logging.error(f"Error fetching prompt templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch prompt templates")



# templates = await build_prompt_templates(gpt_id)
# searching_orders_template = templates["SEARCHING_ORDERS"]

