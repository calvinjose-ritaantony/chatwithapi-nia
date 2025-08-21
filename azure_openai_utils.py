import io
from typing import Any, List

from mongo_service import get_usecases_list

from fpdf import FPDF
from data.NiaTool import NiaTool
from data.useCaseSpecificOutputs import StructuredSpendingAnalysis

import os
import uuid
import base64
import logging
import json
import datetime
import tiktoken
from PIL import Image

from fastapi.responses import StreamingResponse
from fastapi import UploadFile, WebSocket
from openai import APIConnectionError, AsyncAzureOpenAI, AzureOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

from ConnectionManager import ConnectionManager
from data.GPTData import GPTData
from data.ModelConfiguration import ModelConfiguration
from dependencies import NiaAzureOpenAIClient

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.search.documents import SearchClient

from dependencies import NiaAzureOpenAIClient
from gpt_utils import extract_json_content, extract_response, get_previous_context_conversations, get_token_count, handle_upload_files
from standalone_programs.image_analyzer import analyze_image
from dotenv import load_dotenv # For environment variables (recommended)

from mongo_service import fetch_chat_history, delete_chat_history, update_message, get_usecases
from role_mapping import ALL_FIELDS, DEFAULT_MODEL_CONFIGURATION, FORMAT_RESPONSE_AS_MARKDOWN, FUNCTION_CALLING_USER_MESSAGE, IMAGE_ANALYSIS_SYSTEM_PROMPT, NIA_FINOLEX_PDF_SEARCH_SEMANTIC_CONFIGURATION_NAME, NIA_FINOLEX_SEARCH_INDEX, NIA_SEMANTIC_CONFIGURATION_NAME,  CONTEXTUAL_PROMPT, SUMMARIZE_MODEL_CONFIGURATION, USE_CASES_LIST, FUNCTION_CALLING_SYSTEM_MESSAGE, schema_string_spending_pattern #USE_CASE_CONFIG,
from standalone_programs.simple_gpt import run_conversation, ticket_conversations, get_conversation
from routes.ilama32_routes import chat2
from constants import ALLOWED_DOCUMENT_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from thinking_process import REASONING_DATA
from tool_utils import azure_ai_search_tool, web_search_tool, write_response_to_pdf_tool
from web_search_utils import search_web_with_sonar
from prompts import BALANCED_WEB_SEARCH_INTEGRATION, WEB_SEARCH_KEYWORD_CONSTRUCTION_SYSTEM_PROMPT, WEB_SEARCH_KEYWORD_CONSTRUCTION_USER_PROMPT, WEB_SEARCH_DATA_SUMMARIZATION_SYSTEM_PROMPT, SYSTEM_SAFETY_MESSAGE

# from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

load_dotenv()  # Load environment variables from .env file

# Create a logger for this module
logger = logging.getLogger(__name__)

token_encoder = tiktoken.encoding_for_model("gpt-4o") 

# Model = should match the deployment name you chose for your model deployment
delimiter = "```"
DEFAULT_RESPONSE = "N/A"
search_endpoint = os.getenv("SEARCH_ENDPOINT_URL")
search_key = os.getenv("SEARCH_KEY")
search_index = os.getenv("SEARCH_INDEX_NAME")
review_bytes_index = os.getenv("NIA_REVIEW_BYTES_INDEX_NAME")
nia_semantic_configuration_name = os.getenv("NIA_SEMANTIC_CONFIGURATION_NAME")

# Azure Open AI - Model parameters
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
ECOMMERCE_MODEL_NAME = os.getenv("ECOMMERCE_MODEL_NAME")
AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_MODEL_API_VERSION = os.getenv("API_VERSION")

# Azure GPT 4o parameters
GPT_4o_MODEL_NAME = os.getenv("GPT4O_MODEL_NAME")
GPT_4o_2_MODEL_NAME = os.getenv("GPT4O_2_MODEL_NAME")
GPT_4o_API_KEY=os.getenv("GPT4O_API_KEY")

GPT_4o_ENDPOINT_URL=os.getenv("GPT4O_ENDPOINT_URL")

GPT_4o_API_VERSION = os.getenv("GPT4O_API_VERSION")

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group_name = os.getenv("RESOURCE_GROUP_NAME")
openai_account_name = os.getenv("OPENAI_ACCOUNT_NAME")
#previous_conversations_count = os.getenv("PREVIOUS_CONVERSATIONS_TO_CONSIDER")

# Azure Blob Storage - Used for storing image uploads
AZURE_BLOB_STORAGE_CONNECTION_STRING=os.getenv("BLOB_STORAGE_CONNECTION_STRING")
AZURE_BLOB_STORAGE_CONTAINER=os.getenv("BLOB_STORAGE_CONTAINER_NAME")
AZURE_BLOB_STORAGE_ACCOUNT_NAME=os.getenv("BLOB_STORAGE_ACCOUNT_NAME")
AZURE_BLOB_STORAGE_ACCESS_KEY=os.getenv("BLOB_STORAGE_ACCESS_KEY")

DEFAULT_ERROR_RESPONSE_FROM_MODEL="The requested information is not available in the retrieved data. Please try another query or topic."
DEFAULT_FOLLOW_UP_QUESTIONS = ["I would like to know more about this topic", "I need further clarification", "Rephrase your findings"]

blob_service_client = BlobServiceClient(f"https://{AZURE_BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=AZURE_BLOB_STORAGE_ACCESS_KEY
)

USE_CASE_CONFIG = {}

async def getAzureOpenAIClient(azure_endpoint: str, api_key: str, api_version: str, stream: bool) -> AsyncAzureOpenAI:
    #logger.info(f"delimiter: {delimiter} \ndefault_model_name: {default_model_name} \necomm_model_name: {ecomm_model_name} \nazure_endpoint: {azure_endpoint} \napi_key: {api_key} \napi_version: {api_version} \nsearch_endpoint: {search_endpoint} \nsearch_key: {search_key} \nsearch_index: {search_index}")
    
    # # Establish connection to Azure Open AI
    # if stream:
    #     client = AsyncAzureOpenAI(
    #         azure_endpoint=azure_endpoint,
    #         api_key=api_key,
    #         api_version=api_version)
    # else:
    #     client = AzureOpenAI(
    #     azure_endpoint=azure_endpoint,
    #     api_key=api_key,
    #     api_version=api_version)

    # Create the singleton instance
    nia_azure_client: NiaAzureOpenAIClient = await NiaAzureOpenAIClient().create()

    # Retrieve the client
    client = nia_azure_client.get_azure_client()
    
    return client

def get_azure_search_parameters(search_endpoint: str, index_name: str, search_key: str, role_information: str, index_fields: list):
    extra_body = {
        "data_sources": [{
            "type": "azure_search",
            "parameters": {
                "authentication": {
                    "type": "api_key",
                    "key": f"{search_key}"
                },
                "endpoint": f"{search_endpoint}",
                "index_name": f"{index_name}",
                "fields_mapping": {
                    "content_fields_separator": "\n",
                    "content_fields": index_fields,
                    "filepath_field": None,
                    "title_field": None,
                    "url_field": None,
                    "vector_fields": []
                },
                "filter": None,
                "in_scope": True,
                "top_n_documents": 10,
                #"strictness": 3,
                "query_type": "semantic",
                "semantic_configuration": NIA_SEMANTIC_CONFIGURATION_NAME, #"default",
                #"role_information": instructions,
                "role_information": role_information
            }
        }]
    }
    
    logger.info(f"Extra Body for Azure Search: {extra_body}")
    return extra_body

async def store_to_blob_storage(uploadedImage: UploadFile = None):
    try:
        file_name = uploadedImage.filename

        # Initialize Blob Service Client
        blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_STORAGE_CONTAINER, blob=file_name)

        # Upload image to Azure Blob Storage
        blob_client.upload_blob(uploadedImage.file, overwrite=True)

        # Generate a SAS token (valid for 60 minutes)
        sas_token = generate_blob_sas(
            account_name=AZURE_BLOB_STORAGE_ACCOUNT_NAME,
            container_name=AZURE_BLOB_STORAGE_CONTAINER,
            blob_name=file_name,
            account_key=AZURE_BLOB_STORAGE_ACCESS_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=60)
        )

        # Generate URL
        #blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{azure_blob_storage_container}/{file_name}"

        # Generate the full URL with SAS token
        blob_url_with_sas = f"{blob_client.url}?{sas_token}"
        logger.info(f"Blob URL: {blob_url_with_sas}")
        return blob_url_with_sas
    except Exception as e:
        logger.error(f"Error occurred while storing image to Azure Blob Storage: {e}", exc_info=True)
        return DEFAULT_RESPONSE

async def get_data_from_web_search(search_query: str, gpt: GPTData, region: str = "IN", socket_manager: ConnectionManager = None, websocket: WebSocket = None):

    search_summary = "No Data from Web Search"
    logger.info(f"Running web search: query={search_query}, region={region}")
    usecase: str = await get_use_case(gpt)

    if usecase != "DEFAULT" and socket_manager is not None:
        logger.info(f"Performing Web Search  {usecase}")
        await socket_manager.send_json({"response": f"Performing Web Search", "type": "thinking"}, websocket)

    if search_query is None or search_query.strip() == "":
        return search_summary

    try:
        search_summary = await search_web_with_sonar(query=search_query)
        logger.info(f"Web Search Summary {search_summary}")
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)

    return search_summary

async def get_completion_from_messages_standard(user_query: str, gpt: GPTData, model_configuration: ModelConfiguration, conversations: list, use_case: str, use_case_config: dict,  role_information: str, websocket: WebSocket = None, socket_manager: ConnectionManager = None):
    model_response = "No Response from Model"
    main_response = ""
    total_tokens = 0
    follow_up_questions = []
    reasoning = ""

    # This client is synchronous and doesn't need await signal. Set stream=False

    try:
        # Get Azure Open AI Client and fetch response
        client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, False)
        model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)
        extra_body = {}
        
        if gpt["use_rag"] == True:
            logger.info("Assigning additional search parameters for E-commerce model")
            if use_case == "REVIEW_BYTES":
                #extra_body = get_azure_search_parameters(search_endpoint, review_bytes_index, search_key, role_information, review_bytes_index_fields)
                pass
            else:
                #extra_body = get_azure_search_parameters(search_endpoint, search_index, search_key, role_information, ecomm_rag_demo_index_fields)
                pass

        response: ChatCompletion = await client.chat.completions.create(
            model=gpt["name"],
            messages=conversations,
            max_tokens=model_configuration.max_tokens, #max_tokens is now deprecated with o1 models
            temperature=model_configuration.temperature,
            top_p=model_configuration.top_p,
            frequency_penalty=model_configuration.frequency_penalty,
            presence_penalty=model_configuration.presence_penalty,
            extra_body=extra_body,
            seed=100,
            stop=None,
            stream=False,
            user=gpt["user"]
            #n=2,
            #reasoning_effort="low", # available for o1,o3 models only
            #timeout=30,
            #service_tier="auto" # default, flex 
        )
        model_response = response.choices[0].message.content
        logger.info(f"Full Model Response is {response}")

        await do_post_response_processing(user_query=user_query, 
                                          gpt=gpt, 
                                          model_configuration=model_configuration, 
                                          use_case=use_case, 
                                          model_response=model_response, 
                                          use_case_config=use_case_config, 
                                          socket_manager=socket_manager, 
                                          websocket=websocket)

        
        if model_response is None or model_response == "":
            main_response = "No Response from Model. Please try again."
        else:            
            main_response, follow_up_questions, total_tokens = await extract_json_content(response)
            # Fix: If main_response is empty after extracting JSON, use the original model_response
            if not main_response and model_response:
                main_response = model_response
    
    except (APIConnectionError) as retryable_ex:
        logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
        logger.info(f"Retrying with next endpoint")
        
        client: AsyncAzureOpenAI = await NiaAzureOpenAIClient().retry_with_next_endpoint()

        if socket_manager is not None:
            await socket_manager.send_json({"response" : f"Encountered APIConnectionError : Retrying with next endpoint. <br> {client._azure_endpoint} ", "type": "thinking"}, websocket)
        
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=gpt["name"],
                messages=conversations,
                max_tokens=model_configuration.max_tokens,
                temperature=model_configuration.temperature,
                top_p=model_configuration.top_p,
                frequency_penalty=model_configuration.frequency_penalty,
                presence_penalty=model_configuration.presence_penalty,
                extra_body=extra_body,
                seed=100,
                stop=None,
                stream=False,
                user=gpt["user"]
            )
            model_response = response.choices[0].message.content
            logger.info(f"Retry Model Response is {response}")

            await do_post_response_processing(user_query=user_query, 
                                          gpt=gpt, 
                                          model_configuration=model_configuration, 
                                          use_case=use_case, 
                                          model_response=model_response, 
                                          use_case_config=use_case_config, 
                                          socket_manager=socket_manager, 
                                          websocket=websocket)

            if model_response is None or model_response == "":
                main_response = "No Response from Model. Please try again."
            else:
                main_response, follow_up_questions, total_tokens = await extract_json_content(response)
                # Fix: If main_response is empty after extracting JSON, use the original model_response
                if not main_response and model_response:
                    main_response = model_response

        except Exception as final_ex:
            logger.error(f"Retry also failed: {final_ex}", exc_info=True)
            total_tokens = len(token_encoder.encode(str(conversations)))
            main_response = f"All Azure OpenAI endpoints failed. Please try again later.\n\n Exception Details : {str(final_ex)}"

            if socket_manager is not None:
                await socket_manager.send_json({"response" : main_response, "type": "thinking"}, websocket)

    except (RateLimitError) as retryable_ex:
        logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
        logger.info(f"Retrying with next models")
        
        # Gather all retry model names from environment variables
        models_to_try = []
        i = 1
        while True:
            model_env_var = f"GPT_RETRY_MODELS_{i}"
            alt_model = os.getenv(model_env_var)
            if not alt_model:
             break
            models_to_try.append(alt_model)
            i += 1

        # Helper to call the model
        async def call_model(client: AsyncAzureOpenAI, model_name, conversations, model_configuration: ModelConfiguration):
            response: ChatCompletion = await client.chat.completions.create(
            model=model_name,
            messages=conversations,
            max_tokens=model_configuration.max_tokens,
            temperature=model_configuration.temperature,
            top_p=model_configuration.top_p,
            frequency_penalty=model_configuration.frequency_penalty,
            presence_penalty=model_configuration.presence_penalty,
            extra_body=extra_body,
            seed=100,
            stop=None,
            stream=False,
            user=gpt["user"]
            )
           
            model_response = response.choices[0].message.content
            logger.info(f"Alternate Model Response is {response}")

            await do_post_response_processing(user_query=user_query, 
                                          gpt=gpt, 
                                          model_configuration=model_configuration, 
                                          use_case=use_case, 
                                          model_response=model_response, 
                                          use_case_config=use_case_config, 
                                          socket_manager=socket_manager, 
                                          websocket=websocket)

            if model_response is None or model_response == "":
                raise ValueError("No Response from Model. Please try again.")
            main_response, follow_up_questions, total_tokens = await extract_json_content(response)
            # Fix: If main_response is empty after extracting JSON, use the original model_response
            if not main_response and model_response:
                main_response = model_response
            return main_response, follow_up_questions, total_tokens

        success = False
        for alt_model in models_to_try:
            try:
                model_response = "No Response from Model"
                main_response = ""
                total_tokens = 0
                follow_up_questions = []
                reasoning = ""
                main_response, follow_up_questions, total_tokens = await call_model(
                    client, alt_model, conversations, model_configuration
                )
                logger.info(f"Succeeded with alternate model '{alt_model}'")
                logger.info(f"Main Response: {main_response}, Follow Up Questions: {follow_up_questions}, Total Tokens: {total_tokens}")
                success = True
                break

            except RateLimitError as alt_rl_ex:
                logger.warning(
                    f"Rate-limit on alternate model '{alt_model}': {alt_rl_ex!s}",
                    exc_info=True,
                )
                continue  # try the next model

            except APIConnectionError as alt_conn_ex:
                logger.warning(
                    f"Connection issue on alternate model '{alt_model}': {alt_conn_ex!s}",
                    exc_info=True,
                )
                break

        # If we exhausted the loop with no success, go to the next subscription
        if not success:
            logger.info("All models on this endpoint exhausted – switching subscription")

            if socket_manager is not None:
                await socket_manager.send_json({"response" : f"Encountered RateLimitError : All models on this endpoint exhausted – switching subscription ", "type": "thinking"}, websocket)

            client: AsyncAzureOpenAI = await NiaAzureOpenAIClient().retry_with_next_endpoint()

            try:
                # Start again with the original model on the new endpoint
                main_response, follow_up_questions, total_tokens = await call_model(
                    client, gpt["name"], conversations, model_configuration
                )

            except Exception as final_ex:
                logger.error("Retry with next subscription also failed", exc_info=True)
                total_tokens = len(token_encoder.encode(str(conversations)))
                main_response = (
                    "All Azure OpenAI endpoints failed. Please try again later.\n\n"
                    f"Exception Details: {final_ex}"
                )
        
    except BadRequestError as be:
        logger.error(f"BadRequestError occurred while fetching model response: {be}", exc_info=True)
        total_tokens = len(token_encoder.encode(str(conversations)))
        main_response = f"Bad Request error occurred while reaching to Azure Open AI. \n\n Exception Details : " + be.message
    
    except Exception as e:
        logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
        main_response = f"Error occurred while fetching model response: \n\n" + str(e)
    finally:
         # Log the response to database
        if socket_manager is not None:
            await socket_manager.send_json({"response" : f"Adding the conversation to memory", "type": "thinking"}, websocket)
        await saveAssistantResponse(main_response, gpt, conversations)
        
    return {
        "model_response" : main_response,
        "total_tokens": total_tokens,
        "follow_up_questions": follow_up_questions,
        "reasoning" : reasoning
    }

async def get_completion_from_messages_stream(user_query: str, gpt: GPTData, model_configuration: ModelConfiguration, conversations: list, use_case: str, use_case_config: dict, role_information: str, websocket: WebSocket = None, socket_manager: ConnectionManager = None):
     # This client is asynchronous and needs await signal. Set stream=True
    try:
        # Get Azure Open AI Client and fetch response
        client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, True)
        model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)
        extra_body = {}

        if gpt["use_rag"] == True:
            logger.info("Assigning additional search parameters for E-commerce model")
            if use_case == "REVIEW_BYTES":
                #extra_body = get_azure_search_parameters(search_endpoint, review_bytes_index, search_key, role_information, review_bytes_index_fields)
                pass
            else:
                #extra_body = get_azure_search_parameters(search_endpoint, search_index, search_key, role_information, ecomm_rag_demo_index_fields)
                pass
        
        full_response_content = ""
        
        async def stream_processor():
            nonlocal full_response_content
            nonlocal client

            try:
                response: ChatCompletion = await client.chat.completions.create(
                    model=gpt["name"],
                    messages=conversations,
                    max_tokens=model_configuration.max_tokens,
                    temperature=model_configuration.temperature,
                    top_p=model_configuration.top_p,
                    frequency_penalty=model_configuration.frequency_penalty,
                    presence_penalty=model_configuration.presence_penalty,
                    stop=None,
                    stream=True,
                    extra_body=extra_body,
                    seed=100,
                    user=gpt["user"]
                )

                async for chunk in response:
                    if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                        chunkContent = chunk.choices[0].delta.content
                        if chunkContent is not None:
                            full_response_content += chunkContent
                            yield chunkContent

            except APIConnectionError as retryable_ex:
                logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
                logger.info(f"Retrying with next endpoint")
                client = await NiaAzureOpenAIClient().retry_with_next_endpoint()
                try:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=gpt["name"],
                        messages=conversations,
                        max_tokens=model_configuration.max_tokens,
                        temperature=model_configuration.temperature,
                        top_p=model_configuration.top_p,
                        frequency_penalty=model_configuration.frequency_penalty,
                        presence_penalty=model_configuration.presence_penalty,
                        extra_body=extra_body,
                        seed=100,
                        stop=None,
                        stream=True,
                        user=gpt["user"]
                    )

                    async for chunk in response:
                        if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                            chunkContent = chunk.choices[0].delta.content
                            if chunkContent is not None:
                                full_response_content += chunkContent
                                yield chunkContent
                except Exception as final_ex:
                    logger.error(f"Retry also failed: {final_ex}", exc_info=True)
                    full_response_content = f"All Azure OpenAI endpoints failed. Please try again later.\n\n Exception Details : {str(final_ex)}"
                    yield full_response_content

            except RateLimitError as retryable_ex:
                logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
                logger.info(f"Retrying with next models")
                # Gather all retry model names from environment variables
                models_to_try = []
                i = 1
                while True:
                    model_env_var = f"GPT_RETRY_MODELS_{i}"
                    alt_model = os.getenv(model_env_var)
                    if not alt_model:
                        break
                    models_to_try.append(alt_model)
                    i += 1

                # Helper to call the model
                async def call_model(client: AsyncAzureOpenAI, model_name, conversations, model_configuration: ModelConfiguration):
                    response: ChatCompletion = await client.chat.completions.create(
                        model=model_name,
                        messages=conversations,
                        max_tokens=model_configuration.max_tokens,
                        temperature=model_configuration.temperature,
                        top_p=model_configuration.top_p,
                        frequency_penalty=model_configuration.frequency_penalty,
                        presence_penalty=model_configuration.presence_penalty,
                        extra_body=extra_body,
                        seed=100,
                        stop=None,
                        stream=True,
                        user=gpt["user"]
                    )
                    async for chunk in response:
                        if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                            chunkContent = chunk.choices[0].delta.content
                            if chunkContent is not None:
                                nonlocal full_response_content
                                full_response_content += chunkContent
                                yield chunkContent

                success = False
                for alt_model in models_to_try:
                    try:
                        async for chunk in call_model(client, alt_model, conversations, model_configuration):
                            yield chunk
                        logger.info(f"Succeeded with alternate model '{alt_model}'")
                        success = True
                        break

                    except RateLimitError as alt_rl_ex:
                        logger.warning(
                            f"Rate-limit on alternate model '{alt_model}': {alt_rl_ex!s}",
                            exc_info=True,
                        )
                        continue  # try the next model

                    except APIConnectionError as alt_conn_ex:
                        logger.warning(
                            f"Connection issue on alternate model '{alt_model}': {alt_conn_ex!s}",
                            exc_info=True,
                        )
                        break

                # If we exhausted the loop with no success, go to the next subscription
                if not success:
                    logger.info("All models on this endpoint exhausted – switching subscription")
                    client = await NiaAzureOpenAIClient().retry_with_next_endpoint()
                    try:
                        async for chunk in call_model(client, gpt["name"], conversations, model_configuration):
                            yield chunk
                    except Exception as final_ex:
                        logger.error("Retry with next subscription also failed", exc_info=True)
                        full_response_content = (
                            "All Azure OpenAI endpoints failed. Please try again later.\n\n"
                            f"Exception Details: {final_ex}"
                        )
                        yield full_response_content
        
        # Create a wrapper generator that handles post-stream processing
        async def response_wrapper():
            nonlocal full_response_content
            nonlocal gpt
            try:
                async for chunk in stream_processor():
                    #logger.info(f"chunk data {chunk}")
                    yield chunk
            
            except BadRequestError as be:
                logger.error(f"BadRequestError occurred while fetching model response: {be}", exc_info=True)
                full_response_content = f"Bad Request error occurred while reaching to Azure Open AI. \n\n Exception Details : " + be.message
                yield full_response_content
            
            except Exception as e:
                logger.error(f"Exception occurred while fetching model response: {e}", exc_info=True)
                #yield str(re)
                full_response_content = f"Exception occurred while fetching model response: {str(e)}."
                yield full_response_content
            finally:
                # This block ensures post-stream processing happens after the stream is complete
                if full_response_content is not None:
                    # update the response to database
                    await saveAssistantResponse(full_response_content, gpt, conversations)
        
        return StreamingResponse(response_wrapper(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
        return StreamingResponse(iter([str(e)]), media_type="text/event-stream")

async def get_completion_from_messages_default(model_name: str, use_rag: bool, messages: list, model_configuration: ModelConfiguration):

    model_response = "No Response from Model"

    # Get Azure Open AI Client and fetch response
    client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, False)
    model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)

    try:
        response: ChatCompletion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=model_configuration.max_tokens,
            temperature=model_configuration.temperature,
            top_p=model_configuration.top_p,
            frequency_penalty=model_configuration.frequency_penalty,
            presence_penalty=model_configuration.presence_penalty,
            stop=None,
            stream=False,
            seed=100
        )
        logger.info(f"Default Model Response is {response}")
        model_response = response.choices[0].message.content
        #logger.info(f"Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
        model_response = str(e)
    
    return model_response

async def analyzeImage_standard(gpt: GPTData, conversations, model_configuration, save_response_to_db: bool):
    model_response = "No Response from Model"
    main_response = "No Response from Model"
    total_tokens = 0
    follow_up_questions = []

    # Get Azure Open AI Client and fetch response
    #client = getAzureOpenAIClient(gpt4o_endpoint, gpt4o_api_key, gpt4o_api_version, False)
    try:
        client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, False)
        model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)

        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=GPT_4o_2_MODEL_NAME,
                messages=conversations,
                max_tokens=model_configuration.max_tokens,
                temperature=model_configuration.temperature,
                top_p=model_configuration.top_p,
                frequency_penalty=model_configuration.frequency_penalty,
                presence_penalty=model_configuration.presence_penalty,
                stop=None,
                stream=False,
                seed=100
            )
            model_response = response.choices[0].message.content
            #logger.info(f"Model Response is {response}")
            #logger.info(f"Tokens used: {response.usage.total_tokens}")

            if model_response is None or model_response == "":
                model_response = "No Response from Model. Please try again."
            else:
                main_response, follow_up_questions, total_tokens = await processResponse(response)

            # Log the response to database
            if save_response_to_db:
                await saveAssistantResponse(main_response, gpt, conversations)

        except Exception as e:
            logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
            main_response = str(e)

    except (APIConnectionError) as retryable_ex:
        logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)

        from dependencies import NiaAzureOpenAIClient
        logger.info(f"Retrying with next endpoint")
        client = await NiaAzureOpenAIClient().retry_with_next_endpoint()
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=gpt["name"],
                messages=conversations,
                max_tokens=model_configuration.max_tokens,
                temperature=model_configuration.temperature,
                top_p=model_configuration.top_p,
                frequency_penalty=model_configuration.frequency_penalty,
                presence_penalty=model_configuration.presence_penalty,
                # extra_body=extra_body,
                seed=100,
                stop=None,
                stream=False,
                user=gpt["user"]
            )
            model_response = response.choices[0].message.content
            logger.info(f"Retry Model Response is {response}")

            if model_response is None or model_response == "":
                main_response = "No Response from Model. Please try again."
            else:
                main_response, follow_up_questions, total_tokens = await extract_json_content(response)

        except Exception as final_ex:
            logger.error(f"Retry also failed: {final_ex}", exc_info=True)
            total_tokens = len(token_encoder.encode(str(conversations)))
            main_response = f"All Azure OpenAI endpoints failed. Please try again later.\n\n Exception Details : {str(final_ex)}"

    except (RateLimitError) as retryable_ex:
        logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)

        from dependencies import NiaAzureOpenAIClient
        logger.info(f"Retrying with next models")
        # Gather all retry model names from environment variables
        models_to_try = []
        i = 1
        while True:
            model_env_var = f"GPT_RETRY_MODELS_{i}"
            alt_model = os.getenv(model_env_var)
            if not alt_model:
                break
            models_to_try.append(alt_model)
            i += 1

        # Helper to call the model
        async def call_model(client: AsyncAzureOpenAI, model_name, conversations, model_configuration: ModelConfiguration):
            response: ChatCompletion = await client.chat.completions.create(
                model=model_name,
                messages=conversations,
                max_tokens=model_configuration.max_tokens,
                temperature=model_configuration.temperature,
                top_p=model_configuration.top_p,
                frequency_penalty=model_configuration.frequency_penalty,
                presence_penalty=model_configuration.presence_penalty,
                # extra_body=extra_body,
                seed=100,
                stop=None,
                stream=False,
                user=gpt["user"]
            )
            model_response = response.choices[0].message.content
            if model_response is None or model_response == "":
                raise ValueError("No Response from Model. Please try again.")
            return await extract_json_content(response)

        success = False
        for alt_model in models_to_try:
            try:
                main_response, follow_up_questions, total_tokens = await call_model(
                    client, alt_model, conversations, model_configuration
                )
                logger.info(f"Succeeded with alternate model '{alt_model}'")
                success = True
                break

            except RateLimitError as alt_rl_ex:
                logger.warning(
                    f"Rate-limit on alternate model '{alt_model}': {alt_rl_ex!s}",
                    exc_info=True,
                )
                continue  # try the next model

            except APIConnectionError as alt_conn_ex:
                logger.warning(
                    f"Connection issue on alternate model '{alt_model}': {alt_conn_ex!s}",
                    exc_info=True,
                )
                break

        # If we exhausted the loop with no success, go to the next subscription
        if not success:
            logger.info("All models on this endpoint exhausted – switching subscription")
            client = await NiaAzureOpenAIClient().retry_with_next_endpoint()

            try:
                # Start again with the original model on the new endpoint
                main_response, follow_up_questions, total_tokens = await call_model(
                    client, gpt["name"], conversations, model_configuration
                )

            except Exception as final_ex:
                logger.error("Retry with next subscription also failed", exc_info=True)
                total_tokens = len(token_encoder.encode(str(conversations)))
                main_response = (
                    "All Azure OpenAI endpoints failed. Please try again later.\n\n"
                    f"Exception Details: {final_ex}"
                )
    
    except BadRequestError as be:
        logger.error(f"BadRequestError occurred while fetching model response: {be}", exc_info=True)
        total_tokens = len(token_encoder.encode(str(conversations)))
        main_response = f"Bad Request error occurred while reaching to Azure Open AI. \n\n Exception Details : " + be.message
    
    except Exception as e:
        logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
        main_response = f"Error occurred while fetching model response: \n\n" + str(e)
    return {
        "model_response": main_response,
        "total_tokens": total_tokens,
        "follow_up_questions": follow_up_questions
    }

async def analyzeImage_stream(gpt: GPTData, conversations, model_configuration, save_response_to_db: bool):
    # Get Azure Open AI Client and fetch response
    #client = getAzureOpenAIClient(gpt4o_endpoint, gpt4o_api_key, gpt4o_api_version, True)
    try:
        client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, True)
        model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)

        try:
            full_response_content = ""
            
            async def stream_processor():
                nonlocal full_response_content
                nonlocal client
                
                try:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=GPT_4o_2_MODEL_NAME,
                        messages=conversations,
                        max_completion_tokens=model_configuration.max_tokens,
                        temperature=model_configuration.temperature,
                        top_p=model_configuration.top_p,
                        frequency_penalty=model_configuration.frequency_penalty,
                        presence_penalty=model_configuration.presence_penalty,
                        stop=None,
                        stream=True,
                        seed=100
                    )
                    
                    async for chunk in response:
                        nonlocal full_response_content

                        if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                            logger.info(f"chunk.choices[0].delta {chunk.choices[0].delta}")
                            chunkContent = chunk.choices[0].delta.content
                            if chunkContent is not None:
                                full_response_content += chunkContent
                                yield chunkContent
                except APIConnectionError as retryable_ex:
                    logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
                    logger.info(f"Retrying with next endpoint")
                    client = await NiaAzureOpenAIClient().retry_with_next_endpoint()
                    try:
                        response: ChatCompletion = await client.chat.completions.create(
                            model=gpt["name"],
                            messages=conversations,
                            max_tokens=model_configuration.max_tokens,
                            temperature=model_configuration.temperature,
                            top_p=model_configuration.top_p,
                            frequency_penalty=model_configuration.frequency_penalty,
                            presence_penalty=model_configuration.presence_penalty,
                            stop=None,
                            stream=True,
                            seed=100,
                            user=gpt["user"]
                        )
                        async for chunk in response:
                            if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                                chunkContent = chunk.choices[0].delta.content
                                if chunkContent is not None:
                                    full_response_content += chunkContent
                                    yield chunkContent
                    except Exception as final_ex:
                        logger.error(f"Retry also failed: {final_ex}", exc_info=True)
                        full_response_content = f"All Azure OpenAI endpoints failed. Please try again later.\n\n Exception Details : {str(final_ex)}"
                        yield full_response_content

                except RateLimitError as retryable_ex:
                    logger.warning(f"Retryable error: {type(retryable_ex).__name__} - {retryable_ex}", exc_info=True)
                    logger.info(f"Retrying with next models")
                    # Gather all retry model names from environment variables
                    models_to_try = []
                    i = 1
                    while True:
                        model_env_var = f"GPT_RETRY_MODELS_{i}"
                        alt_model = os.getenv(model_env_var)
                        if not alt_model:
                            break
                        models_to_try.append(alt_model)
                        i += 1

                    # Helper to call the model
                    async def call_model(client, model_name, conversations, model_configuration):
                        response: ChatCompletion = await client.chat.completions.create(
                            model=model_name,
                            messages=conversations,
                            max_tokens=model_configuration.max_tokens,
                            temperature=model_configuration.temperature,
                            top_p=model_configuration.top_p,
                            frequency_penalty=model_configuration.frequency_penalty,
                            presence_penalty=model_configuration.presence_penalty,
                            stop=None,
                            stream=True,
                            seed=100,
                            user=gpt["user"]
                        )
                        async for chunk in response:
                            if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                                chunkContent = chunk.choices[0].delta.content
                                if chunkContent is not None:
                                    nonlocal full_response_content
                                    full_response_content += chunkContent
                                    yield chunkContent

                    success = False
                    for alt_model in models_to_try:
                        try:
                            async for chunk in call_model(client, alt_model, conversations, model_configuration):
                                yield chunk
                            logger.info(f"Succeeded with alternate model '{alt_model}'")
                            success = True
                            break

                        except RateLimitError as alt_rl_ex:
                            logger.warning(
                                f"Rate-limit on alternate model '{alt_model}': {alt_rl_ex!s}",
                                exc_info=True,
                            )
                            continue  # try the next model

                        except APIConnectionError as alt_conn_ex:
                            logger.warning(
                                f"Connection issue on alternate model '{alt_model}': {alt_conn_ex!s}",
                                exc_info=True,
                            )
                            break

                    # If we exhausted the loop with no success, go to the next subscription
                    if not success:
                        logger.info("All models on this endpoint exhausted – switching subscription")
                        client = await NiaAzureOpenAIClient().retry_with_next_endpoint()

                        try:
                            async for chunk in call_model(client, gpt["name"], conversations, model_configuration):
                                yield chunk
                        except Exception as final_ex:
                            logger.error("Retry with next subscription also failed", exc_info=True)
                            full_response_content = (
                                "All Azure OpenAI endpoints failed. Please try again later.\n\n"
                                f"Exception Details: {final_ex}"
                            )
                            yield full_response_content
            
            # Create a wrapper generator that handles post-stream processing
            async def response_wrapper():
                nonlocal full_response_content

                try:
                    async for chunk in stream_processor():
                        yield chunk

                except BadRequestError as be:
                    logger.error(f"BadRequestError occurred while fetching model response: {be}", exc_info=True)
                    full_response_content = f"Bad Request error occurred while reaching to Azure Open AI. \n\n Exception Details : " + be.message
                    yield full_response_content

                except Exception as e:
                    logger.error(f"Exception occurred while fetching model response: {e}", exc_info=True)
                    #yield str(re)
                    full_response_content = f"Exception occurred while fetching model response: {str(e)}."
                    yield full_response_content
                finally:
                    # This block ensures post-stream processing happens after the stream is complete
                    if full_response_content is not None:
                        nonlocal gpt
                        # update the response to database
                        if save_response_to_db:
                            await saveAssistantResponse(full_response_content, gpt, conversations)
            
            return StreamingResponse(response_wrapper(), media_type="text/event-stream")
        
        except Exception as e:
            logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
            return StreamingResponse(iter([str(e)]), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error occurred while fetching model response: {e}", exc_info=True)
        return StreamingResponse(iter([str(e)]), media_type="text/event-stream")
    
async def preprocessForRAG(user_message: str, image_response:str, use_case:str, gpt: GPTData, conversations: list, model_configuration: ModelConfiguration, USE_CASE_CONFIG: dict, socket_manager: ConnectionManager = None, websocket: WebSocket = None):

    logger.info(f"USE_CASE : {use_case}")
    USER_PROMPT = USE_CASE_CONFIG[use_case]["user_message"]
    #logger.info(f"USE_CASE_CONFIG[{use_case}]: {USER_PROMPT}")

    #context_information, additional_context_information, conversations = await determineFunctionCalling(user_message, image_response, use_case, gpt, conversations, model_configuration, "pre_response")
    context_information, additional_context_information, web_search_results, conversations = await determineFunctionCalling(search_query=user_message, 
                                                                                                                            image_response=image_response, 
                                                                                                                            use_case=use_case, 
                                                                                                                            gpt=gpt, 
                                                                                                                            conversations=conversations, 
                                                                                                                            model_configuration=model_configuration,
                                                                                                                            scenario="pre_response",
                                                                                                                            use_case_config=USE_CASE_CONFIG,
                                                                                                                            socket_manager=socket_manager, 
                                                                                                                            websocket=websocket)
    
    if web_search_results is None or web_search_results == "":
        web_search_results = "No Data from Web Search"
        
    # Step 4: Append the current user query with additional context into the conversation. This additional context is only to generate the response from the model and won't be saved in the conversation history for aesthetic reasons.
    if use_case == "CREATE_PRODUCT_DESCRIPTION":
        conversations.append({
                                "role": "user",
                                "content" : USER_PROMPT.format(
                                                query=user_message, 
                                                sources=image_response,
                                                web_search_results=web_search_results, 
                                                additional_sources=[]) + FORMAT_RESPONSE_AS_MARKDOWN
                            })
    elif use_case == "ANALYZE_SPENDING_PATTERNS":
        conversations.append({
                                "role": "user",
                                "content" : USER_PROMPT.format(
                                                query=user_message, 
                                                sources=context_information, 
                                                web_search_results=web_search_results, 
                                                additional_sources=additional_context_information,
                                                schema_string_spendingPattern = schema_string_spending_pattern)
                            })
    elif context_information is not None and context_information != "" and len(context_information) > 0:
        conversations.append({
                                "role": "user",
                                "content" : USER_PROMPT.format(
                                                query=user_message, 
                                                sources=context_information, 
                                                web_search_results=web_search_results, 
                                                additional_sources=additional_context_information) + FORMAT_RESPONSE_AS_MARKDOWN
                            })
    else:
        conversations.append({"role": "user", "content": user_message})
    
    token_data = await get_token_count(gpt["name"], gpt["instructions"],  conversations, user_message, int(model_configuration.max_tokens))
    logger.info(f"Token Calculation : stage 1.3 - preprocessForRAG {token_data}")

async def processImage(streaming_response: bool, save_response_to_db: bool, user_message: str, model_configuration: ModelConfiguration, gpt: GPTData, conversations: list, uploadedImage: UploadFile = None, socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    image_url = ""
    base64_image = ""
    try:
        if uploadedImage is not None and uploadedImage.filename != "blob" and uploadedImage.filename != "dummy":
            image_url = await store_to_blob_storage(uploadedImage)

            if image_url == None or image_url == "" or image_url == "N/A":
                logger.info(f"Image URL is empty. Passing Base64 encoded image for inference {image_url}")
                # 1. Read image content
                contents = await uploadedImage.read()

                # 2. Encode as base64 
                base64_image = base64.b64encode(contents).decode('utf-8')
                logger.info(f"Image size (bytes): {len(contents)}")

                image_url = f"data:image/jpeg;base64,{base64_image}"

            logger.info(f"Image URL before sending to model is {image_url}")

            # 3. Prepare messages with image_url
            image_message = {
                                "role": "user", 
                                "content": [
                                    #{"type": "text", "text": user_message},
                                    {
                                        "type": "image_url", 
                                        "image_url": {"url": image_url}
                                    }
                                ]
                            }
            
            # 4. Add image message to the conversation
            conversations.append(image_message)

            await update_message({
                "gpt_id": gpt["_id"],
                "gpt_name": gpt["name"],
                "role": "user",
                #"content": f"data:image/jpeg;base64,{base64_image}"
                "content": image_url,
                "user": gpt["user"],
                "use_case_id": gpt["use_case_id"]
            })

            if await get_use_case(gpt) != "DEFAULT" and socket_manager is not None:
                await socket_manager.send_json({"response" :"Processing the Image", "type": "thinking"}, websocket)

            token_data = await get_token_count(gpt["name"], gpt["instructions"],  conversations, user_message, int(model_configuration.max_tokens))
            logger.info(f"Token Calculation : stage 1.1 - Image analysis {token_data}")

            # 5. Call Azure OpenAI API for image analysis
            if streaming_response:
                response = await analyzeImage_stream(gpt, conversations, model_configuration, save_response_to_db)
            else:
                response = await analyzeImage_standard(gpt, conversations, model_configuration, save_response_to_db)

            logger.info(f"Image Response: {response}")

    except Exception as e:
        logger.error(f"Error occurred while processing image: {e}", exc_info=True)
        response = str(e) # Return the error message as response
    
    return response

async def processResponse(response):
    total_tokens = response.usage.total_tokens
    follow_up_questions=[]
    model_response = response.choices[0].message.content
    main_response = ""

    if model_response is not None and model_response != "" and model_response.find("follow_up_questions") != -1: 
        try:
             response_json = await extract_response(model_response) # Extract the JSON response from the model response. The model response is expected to be wrapped in triple backticks
             main_response = response_json["model_response"]
             follow_up_questions = response_json["follow_up_questions"]

            #main_response, follow_up_questions = extract_response_from_markdown(model_response)
        except Exception as e:
            logger.error(f"Error occurred while processing model response: {e}", exc_info=True)
            main_response = model_response
            follow_up_questions = [] # do not send follow-up questions in exception scenarios
    else:
        # Handle cases where the follow-up questions are missing
        main_response = model_response
        follow_up_questions = DEFAULT_FOLLOW_UP_QUESTIONS

    return main_response, follow_up_questions, total_tokens

async def generate_response(streaming_response: bool, user_message: str, model_configuration: ModelConfiguration, gpt: GPTData, uploadedFile: UploadFile = None, socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    has_image = False
    previous_conversations_count = 6
    response = {
                    # "response": "No Response from model", 
                    "model_response": "No response from model",
                    "total_tokens": 0, 
                    "follow_up_questions": [""],
                    "type": "chat_response"
                }
    proceed = False
    web_search = True
    model_name = gpt["name"]
    use_rag = bool(gpt["use_rag"])
    image_response = ""
    DEFAULT_IMAGE_RESPONSE = ""

    gpt_id = str(gpt["_id"])
    USE_CASE_CONFIG = await update_use_case_config(gpt_id)
 
    # Step 1 : Get the use case, role information, model configuration parameters
    use_case = await get_use_case(gpt)
    role_information, model_configuration = await get_role_information(use_case, USE_CASE_CONFIG) if use_rag else ("AI Assistant", model_configuration)
    model_configuration: ModelConfiguration = await construct_model_configuration(model_configuration)
 
    # Fetch the thinking process for the selected use case
    if use_case != "DEFAULT":
        thinking_process_for_usecases: list = REASONING_DATA[use_case]
        logger.info(thinking_process_for_usecases)

    if use_case != "DEFAULT" and socket_manager is not None:
        await socket_manager.send_json({"response" : f" Identified the Usecase : {use_case.replace('_', ' ').title()}", "type": "thinking"}, websocket)

    # Step 2 : Get last conversation history (6 messages) for the given gpt_id and model_name
    chat_history = await fetch_chat_history(gpt["_id"], model_name, limit=6) # use limit=-1 if needing the entire conversation history to be passed to the model
   
    # Step 3: Format the conversation to support OpenAI format (System Message, User Message, Assistant Message)
    conversations = [{"role": "system", "content": gpt["instructions"]}]
    for msg in chat_history:
        conversations.append({"role": msg["role"], "content": msg["content"]})
 
    # Step 4: get token count for the conversation
    token_data = await get_token_count(model_name, gpt["instructions"],  conversations, user_message, int(model_configuration.max_tokens))
    logger.info(f"Token Calculation : stage 1 {token_data}")
    
    # Get previous conversation for context
    #previous_conversations = get_previous_context_conversations(conversation_list=conversations, previous_conversations_count=previous_conversations_count)
 
    # Step 5: Add the current user query to the messages Collection (Chat History). Avoid saving the query with additional grounded prompt information
    await update_message({
        "gpt_id": gpt["_id"],
        "gpt_name": gpt["name"],
        "role": "user",
        "content": f"{user_message}",
        "user": gpt["user"],
        "use_case_id": gpt["use_case_id"]
    })

    if isinstance(uploadedFile, str) and (uploadedFile.find("data:image/jpeg;base64") != -1 or uploadedFile.find("data:image/png;base64") != -1): #base64 image from websocket API
        has_image = True
        base64_image = uploadedFile.split(",")[1]
        # Decode the base64 image and create a dummy UploadFile object
        image_data = base64.b64decode(base64_image)
        uploadedFile = UploadFile(filename="current_uploaded_image", file=io.BytesIO(image_data), size=len(image_data))
    
    #has_image = (uploadedFile is not None and uploadedFile.filename != "blob" and uploadedFile.filename != "dummy")
    logger.info(f"Uploaded File {uploadedFile}")
    if uploadedFile is not None and isinstance(uploadedFile, UploadFile):
        if uploadedFile.filename != "blob" and uploadedFile.filename != "dummy":
            file_extension = os.path.splitext(uploadedFile.filename)[1].lower()
            logger.info(f"File extension detected: {file_extension}")
            if file_extension in ALLOWED_IMAGE_EXTENSIONS:
                has_image = True
           
            if file_extension in [".pdf"]:
                use_rag = True
                await handle_upload_files(gpt["_id"], gpt, [uploadedFile])
 
    logger.info(f"use_rag is {use_rag} and has_image is {has_image}")
    logger.info(f"Uploaded File {uploadedFile}")

    if use_case != "DEFAULT":
        pass
        # await socket_manager.send_json({"response" : thinking_process_for_usecases[2], "type": "thinking"}, websocket)

    # Step 6: Handle images/attachments if any or the user query
    if not use_rag and has_image:
        logger.info("CASE 1 : No RAG but Image is present")
        proceed = True     
        # Step 1 : Process the image
        image_result = await processImage(streaming_response, True, user_message, model_configuration, gpt, conversations, uploadedFile,socket_manager, websocket)
        
        # Step 2 : Pass the image response to the function calling to determine the next steps
        context_information, additional_context_information, web_search_results, conversations = await determineFunctionCalling(
            search_query=user_message,      # The user's text query
            image_response=image_result,      # Pass the image analysis result as context
            use_case=use_case,          # Your current use case string
            gpt=gpt,
            conversations=conversations,
            model_configuration=model_configuration,
            scenario="post_response",
            use_case_config=USE_CASE_CONFIG,
            socket_manager=socket_manager, 
            websocket=websocket)
           
        if web_search_results is None or web_search_results == "":
            web_search_results = "No Data from Web Search"

        USER_PROMPT = USE_CASE_CONFIG.get(use_case, {}).get("user_message", "{query}")

        conversations.append({
            "role": "user",
            "content": USER_PROMPT.format(
                query=user_message,
                sources=image_result,
                web_search=web_search_results,
                additional_sources=additional_context_information
            ) + FORMAT_RESPONSE_AS_MARKDOWN
        })

    elif use_rag and has_image:
        logger.info("CASE 2 : RAG and Image is present")
        proceed = True     
        # Step 1 : Process the image (Always keep the stream flag as False when processing the image with RAG. Because we need full information of the image for the function calling to take a decision. Streaming will cause problems)
        conversation_for_image_analysis = []
 
        # Create a specialized image analysis system message
        conversation_for_image_analysis.append({
            "role": "system",
            "content": IMAGE_ANALYSIS_SYSTEM_PROMPT
        })
       
        image_response = await processImage(False, False, user_message, model_configuration, gpt, conversation_for_image_analysis, uploadedFile, socket_manager, websocket)
        conversation_for_image_analysis.clear()
 
        # Step 2 : Function Calling
        if image_response is not None and image_response.get("model_response") is not None and image_response.get("model_response") != "":
            await preprocessForRAG(user_message, image_response.get("model_response"), use_case, gpt, conversations, model_configuration, USE_CASE_CONFIG, socket_manager, websocket)
    elif use_rag and not has_image:
        logger.info("CASE 3 : RAG and No Image")
        proceed = True
        await preprocessForRAG(user_message, DEFAULT_IMAGE_RESPONSE, use_case, gpt, conversations, model_configuration, USE_CASE_CONFIG, socket_manager, websocket)
        conversations.append({"role": "user", "content": user_message})
    else:
        logger.info("CASE 4 : No RAG and No Image")
        proceed = True
        logger.info("No function calling. Plain query used as user message")

        if web_search:
            web_search_summary = await get_data_from_web_search(search_query=user_message, gpt=gpt, region="IN", socket_manager=socket_manager, websocket=websocket)
            context_information = "No Data"
            conversations.append({"role": "user", "content": BALANCED_WEB_SEARCH_INTEGRATION.format(
                                                user_query=user_message,
                                                contextual_data=context_information,
                                                web_search_results=web_search_summary) + FORMAT_RESPONSE_AS_MARKDOWN})
            logger.info("inside web search call")
            logger.info(f"Web Search Summary: {web_search_summary}")
        else:
            conversations.append({"role": "user", "content": user_message})
        
    # Step 7: Get the token count after the user message is added to the conversation
    token_data = await get_token_count(model_name, gpt["instructions"],  conversations, user_message, int(model_configuration.max_tokens))
    logger.info(f"Token Calculation : stage 2 (Before generating response) {token_data}")

    # Azure OpenAI API call
    if proceed == True:
        if use_case != "DEFAULT" and socket_manager is not None:
            await socket_manager.send_json({"response" : "Send Data to Model for Response Generation", "type": "thinking"}, websocket)

        if streaming_response:
            response = await get_completion_from_messages_stream(user_message, gpt, model_configuration, conversations, use_case, USE_CASE_CONFIG, role_information, websocket, socket_manager)
        else:
            if use_case != "DEFAULT":
                pass
                # await socket_manager.send_json({"response" : thinking_process_for_usecases[4], "type": "thinking"}, websocket)
            response = await get_completion_from_messages_standard(user_message, gpt, model_configuration, conversations, use_case, USE_CASE_CONFIG, role_information, websocket, socket_manager)
            logger.info(f"Response from model: {response}")

    # Sometimes model returns "null" which is not supported by python
    # the null gets into the chat history and ruins all the subsequent calls to the model
    # hence, we need this check
    if response is None:
        response = "No response from model"
    else:
        pass
    
    if socket_manager is not None:
        await socket_manager.send_json({"response" : "Response Generated", "type": "thinking"}, websocket)
       
    logger.info(f"Conversation : {conversations}")
    logger.info(f"Tokens in the conversation {len(token_encoder.encode(str(conversations)))}")
 
    # summarize the conversation history if its close to 90% of the token limit
    # if int(response["total_tokens"]) >= int(model_configuration["max_tokens"]) * 0.9:
    #     conversation_summary = await summarize_conversations(conversations, gpt)
    #     # Insert the summarization into the messages collection with role "assistant"
    #     update_message({
    #         "gpt_id": gpt["_id"],
    #         "gpt_name": gpt["name"],
    #         "role": "assistant",
    #         "content": f"Below is the summary of the previous conversations with the assistant. \n{conversation_summary}"
    #     })
 
    return response


async def get_data_from_azure_search(search_query: str, use_case: str, gpt_id: str, get_extra_data: bool, use_case_config: dict, socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    """
    # PREREQUISITES
        pip install azure-identity
        pip install azure-search-documents
    # USAGE
        python search_documents.py
    """
    logger.info("Inside fetch data from Azure Search")

    sources_formatted = ""
    additional_results_formatted = ""
    
    get_extra_data = bool(get_extra_data)
    logger.info(f"Search Query: {search_query} \nUse Case: {use_case} \nGet Extra Data: {get_extra_data}")

    # Your MSAL app credentials (Client ID, Client Secret, Tenant ID)
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET_VALUE")
    tenant_id = os.getenv("TENANT_ID")

    logger.info(f"Client ID: {client_id} \nClient Secret: {client_secret} \nTenant ID: {tenant_id}")
    use_cases = await get_usecases(gpt_id)

    if use_case != "DEFAULT"  and socket_manager is not None:
        await socket_manager.send_json({"response": "Starting Azure Search for data...", "type": "thinking"}, websocket)

    # Extract the matching use case from the collection
    use_case_data = next((uc for uc in use_cases if uc["name"] == use_case), None)

    index_name = None
    semantic_configuration_name = None
    if use_case_data:
        index_name = use_case_data.get("index_name", None)
        logger.info(f"Index Name found: {index_name}")
        semantic_configuration_name = use_case_data.get("semantic_configuration_name", None)
        logger.info(f"Semantic Configuration Name found: {semantic_configuration_name}")
    else:
        logger.warning(f"No matching Index and Semantic Configuration found for: {use_case}")

    #logger.info(f"use_case: {use_case}")

    try:
         # Use the token with Azure SDK's client
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)

        # Create a search client
        azure_ai_search_client = SearchClient(
            endpoint=os.getenv("SEARCH_ENDPOINT_URL"),
            #index_name=os.getenv("SEARCH_INDEX_NAME"),
            # index_name=USE_CASE_CONFIG[use_case]["index_name"],
            index_name = index_name,
            credential=credential)
        
        if not all([client_id, client_secret, tenant_id, search_endpoint, search_index]):
            raise ValueError("Missing environment variables.")
        
        logger.info(f"Search Client: {azure_ai_search_client} \nSearch Query: {search_query}")

        # Get the documents
        if use_case == "TRACK_ORDERS_TKE" or use_case == "MANAGE_TICKETS" or use_case == "REVIEW_BYTES" or use_case == "COMPLAINTS_AND_FEEDBACK" or use_case == "SEASONAL_SALES" or use_case == "DOC_SEARCH":
            selected_fields = use_case_config[use_case]["fields_to_select"]
        else:
            selected_fields = ALL_FIELDS 

        logger.info(f"Selected Fields: {selected_fields}")
        # semantic_config_name = USE_CASE_CONFIG[use_case]["semantic_configuration_name"]
        # logger.info(f"Semantic Config Name {semantic_config_name}")
        #selected_fields = ["user_name", "order_id", "product_description", "brand", "order_date", "status", "delivery_date"]
        search_results = azure_ai_search_client.search(search_text=search_query, 
                                                 #top = 5,
                                                 top=use_case_config.get(use_case, {}).get("document_count", 30), 
                                                 include_total_count=True, 
                                                 query_type="semantic",
                                                #  semantic_configuration_name=USE_CASE_CONFIG[use_case]["semantic_configuration_name"],
                                                 semantic_configuration_name = semantic_configuration_name,
                                                 select=selected_fields)
        additional_search_results = []
        if get_extra_data:
            logger.info("Fetching additional data from Azure Search")
            # Create a search client
            additional_azure_ai_search_client = SearchClient(
                endpoint=os.getenv("SEARCH_ENDPOINT_URL"),
                index_name = NIA_FINOLEX_SEARCH_INDEX,
                credential=credential
            )

            additional_search_results = additional_azure_ai_search_client.search(search_text=search_query, 
                                                 top=use_case_config.get(use_case, {}).get("document_count", 30), 
                                                 include_total_count=True, 
                                                 query_type="semantic",
                                                 semantic_configuration_name = NIA_FINOLEX_PDF_SEARCH_SEMANTIC_CONFIGURATION_NAME,
                                                 select=["Name_of_Supplier", "Purchase_Order_Number", "Purchase_Order_Date", "Expense_Made_For", "Quantity", "Net_Price", "Total_Expense", "Supplier_Supplying_Plant", "Currency"])
            
            logger.info(f"search endpoint url {NIA_FINOLEX_SEARCH_INDEX}\n semantic config {NIA_FINOLEX_PDF_SEARCH_SEMANTIC_CONFIGURATION_NAME}")
            
            logger.info(f"Additional Context Information: {additional_search_results}")
            additional_results_list = [result for result in additional_search_results]
            additional_results_formatted = json.dumps(additional_results_list, default=lambda x: x.__dict__, indent=2)
            logger.info(f"Additional Context Information: {additional_results_formatted}")
        
        logger.info("Documents in Azure Search:")
        

        # Convert SearchItemPaged to a list of dictionaries
        results_list = [result for result in search_results]

        # Serialize the results
        sources_formatted = json.dumps(results_list, default=lambda x: x.__dict__, indent=2)
        logger.info(f"Context Information: {sources_formatted}")
        
    except Exception as e:
        sources_formatted = ""
        additional_results_formatted = ""
        logger.error(f"Exception while fetching data from Azure Search {str(e)}", exc_info=True)
    
    return sources_formatted, additional_results_formatted

def get_azure_openai_deployments():
    """
    # PREREQUISITES
        pip install azure-identity
        pip install azure-mgmt-cognitiveservices
    # USAGE
        python list_deployments.py

        Before run the sample, please set the values of the client ID, tenant ID and client secret
        of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
        AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
        https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
    """
    logger.info("Inside fetch deployments from Azure")

    try:
        client = CognitiveServicesManagementClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id
        )

        response = client.deployments.list(
            resource_group_name=resource_group_name,
            account_name=openai_account_name,
        )

        logger.info("Deployments in Azure:")
        for item in response:
            logger.info(item)
    except Exception as e:
        logger.error("Exception while fetching deployments from Azure OpenAI", exc_info=True)

async def summarize_conversations(chat_history, gpt):
    """
    Summarize the conversations using LLM (replace with your code)
    """
    logger.info(f"Length of chat history: {len(chat_history)}")

    gpt_name = gpt["name"]

    if chat_history is not None and len(chat_history) > 0:
        summarization_system_prompt = f"""
            You are a text summarizer. 
            Your task is to read, analyze and understand the conversations provided in the triple backticks (```) and summarize into a meaningful text.
        """

        summarization_user_prompt = f"""
            Analyze and rephrase the conversations wrapped in triple backticks (```), 
            Perform a inner monologue to understand each conversation and generate summary capturing the highlights.
            Skip any irrelevant parts in the conversation that doesn't add value.
            The summary should be detailed, informative with all the available factual data statistics and within 800 words.
            
            {delimiter} {json.dumps(chat_history)} {delimiter}
        """

        messages = [
                    {"role": "system", "content": summarization_system_prompt },
                    {"role": "user", "content": summarization_user_prompt }
                   ]

        #model_configuration: ModelConfiguration = ModelConfiguration(**SUMMARIZE_MODEL_CONFIGURATION)
        model_configuration: ModelConfiguration = model_configuration if  isinstance(model_configuration, ModelConfiguration) else ModelConfiguration(**SUMMARIZE_MODEL_CONFIGURATION)

        # Get Azure Open AI Client and fetch response
        conversation_summary = await get_completion_from_messages_default(DEFAULT_MODEL_NAME, messages, model_configuration)

        # Remove the summarized conversations from the messages collection
        delete_chat_history(gpt["_id"], gpt["name"])
        logger.info(f"Deleted chat history (post summarization) for GPT: { gpt_name} successfully.")

    return conversation_summary

async def determineFunctionCalling(search_query: str, image_response: str, use_case: str, gpt: GPTData, conversations: list, model_configuration: ModelConfiguration, scenario: str, use_case_config: dict, socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    function_calling_conversations = []
    data = []
    additional_data = []
    web_search_results = None

    deployment_name = gpt["name"]
    gpt_id: str = str(gpt["_id"]) 

    logger.info(f"determineFunctionCalling calling Start {deployment_name}")

    # Azure Open AI Clients for different tasks
    azure_openai_client =  AsyncAzureOpenAI(
        azure_endpoint=GPT_4o_ENDPOINT_URL, 
        api_key=GPT_4o_API_KEY, 
        api_version=GPT_4o_API_VERSION)
    
    if use_case == "TRACKING_ORDERS_TKE":
        search_query = search_query + "(TKE)"

    

    # Get the tools and tool definitions
    tool_names, tool_definitions = await get_tools(gpt_id=gpt["_id"], use_case=use_case, scenario=scenario)

    # Initial user message
    function_calling_conversations.append({
                                              "role": "system", 
                                              "content":FUNCTION_CALLING_SYSTEM_MESSAGE.format(tools=tool_names)
                                          })
     
    function_calling_conversations.append({
                                            "role": "user", 
                                           "content": FUNCTION_CALLING_USER_MESSAGE.format(query=search_query, 
                                                                                            use_case=use_case, 
                                                                                            conversation_history=conversations, 
                                                                                            image_details=image_response)
                                        }) 
    #messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined

    response_from_function_calling_model = ""
    function_calling_model_response = ""

    try:
        # First API call: Ask the model to use the function
        response_from_function_calling_model = await azure_openai_client.chat.completions.create(
            model=GPT_4o_2_MODEL_NAME,
            messages=function_calling_conversations,
            tools=tool_definitions,
            #tool_choice="none",
            tool_choice="auto",
            #tool_choice={"type": "function", "function" : {"name"  : "get_data_from_azure_search"}}
            seed=200
        )

        logger.info(f"Full function calling response : {response_from_function_calling_model}")

        # Process the model's response
        function_calling_model_response = response_from_function_calling_model.choices[0].message
        #function_calling_conversations.append(response_message)

        # Handle function calls
        if function_calling_model_response.tool_calls:
            for tool_call in function_calling_model_response.tool_calls:
                if tool_call.function.name == "get_data_from_azure_search":
                    logger.info("get_data_from_azure_search called")
                    function_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Function arguments: {function_args}")  
                    data, additional_data = await get_data_from_azure_search(
                        search_query=function_args.get("search_query"),
                        use_case=function_args.get("use_case"),
                        get_extra_data= function_args.get("get_extra_data") if use_case == "DOC_SEARCH" else False, # Only for doc search the fetch of extra data must be enabled
                        use_case_config=use_case_config,
                        gpt_id = gpt_id,
                        socket_manager=socket_manager,
                        websocket=websocket      
                    )
                elif tool_call.function.name == "write_response_to_pdf":
                    logger.info("write_response_to_pdf called")
                    function_args = json.loads(tool_call.function.arguments)
                    logger.info(f"[PDF INTENT] PDF tool called with args: {function_args}")
                    await write_response_to_pdf(
                        pdf_content=function_args.get("response_text", ""),
                        gpt=gpt,
                        file_name=function_args.get("file_name", "nia_response"),
                        socket_manager=socket_manager,
                        websocket=websocket  # Default file name if not provided
                    )
                elif tool_call.function.name == "get_data_from_web_search":
                    logger.info("Calling get_data_from_web_search")
                    function_args = json.loads(tool_call.function.arguments)
                    web_search_results = await get_data_from_web_search(
                        search_query=function_args.get("search_query"),
                        gpt=gpt,
                        region=function_args.get("region", "IN"),
                        socket_manager=socket_manager,
                        websocket=websocket
                    )
                else:
                    logger.info(f"Couldn't find tool {tool_call.function.name} in NIA. Kindly check the tool definition.s")

                    # Append the function response to the original conversation list
                    # conversations.append({
                    #     "tool_call_id": tool_call.id,
                    #     "role": "tool",
                    #     "name": "get_data_from_azure_search",
                    #     "content": data #data, # commenting data because it will be redundant in the conversation history as we are adding the contextual_information to sources in USER_PROMPT
                    # })
            logger.info("Function calling END")
        else:
            logger.info("No tool calls were made by the model.")

    except RateLimitError as rle:
        logger.error(f"RateLimitError occurred while calling the function: {rle}", exc_info=True)
        function_calling_model_response = "ERROR#####" + str(rle) + "Your token utilization is high (Max tokens per window 8000). Please try again later."
    except Exception as e:
        logger.error(f"Error occurred while calling the function: {e}", exc_info=True)
        function_calling_model_response = "ERROR#####" + str(e)
    finally:
        token_data = await get_token_count(gpt["name"], gpt["instructions"],  function_calling_conversations, search_query, int(model_configuration.max_tokens))
        logger.info(f"Token Calculation : stage 1.2 - Function calling {token_data}")
        function_calling_conversations.clear() # Clear the messages list because we do not need the system message, user message in this function

    return data, additional_data, web_search_results, conversations


async def call_maf(ticketId: str):
    client = await getAzureOpenAIClient(AZURE_ENDPOINT_URL, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL_API_VERSION, False)
    model_output = await run_conversation(client, ticketId)
    return model_output

async def get_use_case(gpt: GPTData) -> str:
    """
    Extract the use case from the system message.
    The system message is expected to be in the format: "You are a helpful assistant for <use_case>."
    """

    use_case = "DEFAULT"  # Default value if no use case is found

    if "@@@@@" in gpt["instructions"]:
        instruction_data = gpt["instructions"].split("@@@@@")
        use_case = instruction_data[1]
    else:
        logger.warning("No use case found in the system message.")

    return use_case

##################

async def update_use_case_config(gpt_id: str):
   
    usecases_list = await get_usecases(gpt_id)  # handle null condition 
    USE_CASE_CONFIG = {uc["name"]: uc for uc in usecases_list if "name" in uc}
    return USE_CASE_CONFIG

###############

async def construct_model_configuration(model_configuration) -> ModelConfiguration:
    """
    Construct the model configuration based on the GPT data and provided model configuration.
    """
    model_configuration: ModelConfiguration = model_configuration if  isinstance(model_configuration, ModelConfiguration) else ModelConfiguration(**model_configuration)
    return model_configuration

async def saveAssistantResponse(response: str, gpt: GPTData, conversations: list):
    # Log the response to database
        await update_message({
            "gpt_id": gpt["_id"], # Make sure gpt is accessible here
            "gpt_name": gpt["name"], # Make sure gpt is accessible here
            "role": "assistant",
            "content": response,
            "user": gpt["user"],
            "use_case_id": gpt["use_case_id"],
        })

        conversations.append({"role": "assistant", "content": response}) # Append the response to the conversation history
    
# --- Helper to serialize and write structured response to PDF ---
async def write_response_to_pdf(pdf_content: str, gpt: GPTData, file_name: str = "nia_response", socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    """
    Try to parse as StructuredSpendingAnalysis and pretty print if possible, else fallback to string.
    """
    logger.info("Generating PDF for use case 1")

    try:
        if pdf_content is None or pdf_content == "":
            return
        logger.info("Generating PDF for use case")

        if await get_use_case(gpt) != "DEFAULT" and socket_manager is not None:
            await socket_manager.send_json({"response": "Generating the pdf...", "type": "thinking"}, websocket)
            logger.info("Thinking message sent successfully")
            
        parsed = None
        if isinstance(pdf_content, str):
            try:
                parsed = StructuredSpendingAnalysis.parse_raw(pdf_content)
            except Exception:
                try:
                    parsed = StructuredSpendingAnalysis.parse_obj(json._json.loads(pdf_content))
                except Exception:
                    parsed = None
        if parsed:
            # Serialize to readable text
            text = f"{parsed.title}\n\n{parsed.description}\n\n"
            for block in parsed.blocks:
                if block.block_type == "text":
                    text += f"{block.title}\n{block.content}\n\n"
                elif block.block_type == "table":
                    text += f"{block.title}\n"
                    text += "\t".join(block.headers) + "\n"
                    for row in block.rows:
                        text += "\t".join(row) + "\n"
                    text += "\n"
                elif block.block_type == "chart":
                    text += f"{block.title} ({block.chart_type} chart)\n"
                    text += f"X: {', '.join(block.x)}\nY: {', '.join(map(str, block.y))}\n\n"
            text += f"{parsed.closure}\n"
            pdf_content = text

        # Write the content to PDF file
        await generate_pdf_from_text(pdf_content, f"{file_name}.pdf")
    except Exception as ser_ex:
        logger.error(f"[PDF INTENT] Error serializing structured response: {ser_ex}")
        # Fallback: write to text file if PDF generation fails
        try:
            # Create directory if it doesn't exist
            text_dir = os.path.join("static", "pdfs")
            os.makedirs(text_dir, exist_ok=True)
            
            # Write to text file as fallback
            text_file_path = os.path.join(text_dir, f"{file_name}.txt")
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(pdf_content)
            logger.info(f"[PDF INTENT] Fallback: Wrote content to text file at {text_file_path}")
        except Exception as text_ex:
            logger.error(f"[PDF INTENT] Error writing fallback text file: {text_ex}")

async def generate_pdf_from_text(text: str, file_name: str = "nia_response.pdf") -> str:
    pdf: FPDF = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=1)
    # Get user's Documents folder
    documents_dir = os.path.join(os.path.expanduser("~"), "Documents")
    nia_dir = os.path.join(documents_dir, "NIA")
    date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(nia_dir, date_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Always use the same filename, overwrite if exists
    output_path = os.path.join(output_dir, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path
    
async def get_tools(gpt_id: str, use_case: str, scenario: str):
    tool_definitions = []
    tool_names = []

    # Get all the tools
    tools: List[NiaTool] = [
        await azure_ai_search_tool(gpt_id, use_case),
        await web_search_tool(),
        await write_response_to_pdf_tool()
    ]

    # Filter out the tool definitions based on scenario property (pre or post)
    filtered_tools: List[NiaTool] = [tool for tool in tools if tool.tool_type == scenario]

    # Get the tool name and tool definitions
    tool_definitions: List[object] = [filtered_tool.tool_definition for filtered_tool in filtered_tools]  
    tool_names: List[str] = [filtered_tool.tool_name + "("+ filtered_tool.tool_description +")" for filtered_tool in filtered_tools]

    return tool_names, tool_definitions

async def get_last_user_message(conversations: list):
    """
    Extracts the last message with role 'user' from the conversations list.
    
    Args:
        conversations: A list of message objects with 'role' and 'content' keys
        
    Returns:
        The content of the last user message, or None if no user message is found
    """
    last_user_message = None

    if not conversations or not isinstance(conversations, list):
        return last_user_message
    
    # Iterate through the conversations list in reverse order
    for message in reversed(conversations):
        if isinstance(message, dict) and message.get("role") == "user":
            logger.info(f"Last user message found: {message.get('content')}")
            last_user_message = message.get("content")
    
    return last_user_message

async def call_llm(client: AsyncAzureOpenAI, gpt: GPTData, conversations: List[dict], model_configuration:              ModelConfiguration, extra_body: dict = None):
    response: ChatCompletion = await client.chat.completions.create(
                model=gpt["name"],
                messages=conversations,
                max_tokens=model_configuration.max_tokens, #max_tokens is now deprecated with o1 models
                temperature=model_configuration.temperature,
                top_p=model_configuration.top_p,
                frequency_penalty=model_configuration.frequency_penalty,
                presence_penalty=model_configuration.presence_penalty,
                extra_body=extra_body,
                seed=100,
                stop=None,
                stream=False,
                user=gpt["user"]
                #n=2,
                #reasoning_effort="low", # available for o1,o3 models only
                #timeout=30,
                #service_tier="auto" # default, flex 
            )
    model_response = response.choices[0].message.content
    return response, model_response

async def do_post_response_processing(user_query: str, gpt: GPTData, model_configuration: ModelConfiguration, use_case: str, model_response: str, use_case_config: dict, socket_manager: ConnectionManager = None, websocket: WebSocket = None):
    logger.info("[PDF INTENT] Checking for PDF intent with OpenAI function calling...")
    pdf_intent_conversation = [
        {"role": "user", "content": user_query}, 
        {"role": "assistant", "content": model_response}
    ]
    logger.info(f"[PDF INTENT] Calling OpenAI with tools")
    
    await determineFunctionCalling(search_query=user_query, 
                                   image_response="No Data", 
                                   use_case=use_case, 
                                   gpt=gpt, 
                                   conversations=pdf_intent_conversation, 
                                   model_configuration=model_configuration, 
                                   scenario="post_response", 
                                   use_case_config=use_case_config,
                                   socket_manager=socket_manager, 
                                   websocket=websocket)


def base64_to_image(base64_string: str, save_path=None, filename=None):
    """
    Convert base64 encoded image string to an image file.
    
    Args:
        base64_string: Base64 encoded image string
        save_path: Directory to save the image (optional)
        filename: Name for the saved file (optional)
        
    Returns:
        PIL Image object and file path if saved
    """
    try:
        # Check if the base64 string is empty or None
        if not base64_string:
            logging.error("Empty base64 string provided")
            return None, None
            
        # If the string contains a data URI prefix, remove it
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image if a path is provided
        filepath = None
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Generate a filename if not provided
            if not filename:
                filename = f"{uuid.uuid4()}.{image.format.lower() if image.format else 'png'}"
                
            filepath = os.path.join(save_path, filename)
            image.save(filepath)
            logging.info(f"Image saved to {filepath}")
            
        return image, filepath
        
    except Exception as e:
        logging.error(f"Error converting base64 to image: {e}", exc_info=True)
        return None, None
    
async def get_role_information(use_case, USE_CASE_CONFIG: dict):
    role_information = "e-commerce analytics agent"
    model_configuration = DEFAULT_MODEL_CONFIGURATION

    # Mapping of use cases to role_information values
    if use_case and use_case in USE_CASE_CONFIG:
        role_information = USE_CASE_CONFIG[use_case]["role_information"]
        model_configuration = USE_CASE_CONFIG[use_case]["model_configuration"]

    logger.info(f"Use_case: {use_case} \n Role Information: {role_information} \n Model Configuration: {model_configuration}")
    
    return role_information, model_configuration
