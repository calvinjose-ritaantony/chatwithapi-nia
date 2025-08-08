import os
import json
import logging
import base64
from pathlib import Path
from bson import ObjectId
import urllib.parse
from typing import Dict, List

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.identity import ClientSecretCredential
from azure.core.exceptions import AzureError

from pymongo.errors import DuplicateKeyError
from app_config import RAG_DOCUMENTS_FOLDER
from auth_msal import get_current_user_ws
from data.GPTData import GPTData
from data.ModelConfiguration import ModelConfiguration
from gpt_utils import handle_upload_files, create_folders
from azure_openai_utils import generate_response
from mongo_service import fetch_chat_history_for_use_case, get_gpt_by_id, create_new_gpt, get_gpts_for_user, update_gpt, delete_gpt, delete_gpts, delete_chat_history, fetch_chat_history, get_usecases, update_gpt_instruction, get_collection, get_prompts, update_prompt, delete_prompt, update_usecases
from prompt_utils import PromptValidator
from app_config import socket_manager

from bson import ObjectId
from dotenv import load_dotenv # For environment variables (recommended)

# Initialize global variables
conversations = []
use_cases = []
max_tokens_in_conversation = 10 # To be implemented
max_conversations_to_consider = 10
nia_thinking_process = []


delimiter = "```"
load_dotenv()  # Load environment variables from .env file
create_folders()

# Create a logger for this module
logger = logging.getLogger(__name__)

# Create the router
router = APIRouter()

# WebSocket endpoint for creating a new GPT
@router.websocket("/ws/create_gpt/{user_id}")
async def ws_create_gpt(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                gpt = data.get("gpt", {})
                loggedUser = data.get("loggedUser")

                if loggedUser != None and loggedUser != "N/A":
                    gpt["user"] = loggedUser
                    gpt["use_case_id"] = ""
                    # Now you can access gpt as a dictionary
                    gpt = GPTData(**gpt)  # Validate and create GPTData instance

                    # Handle file upload separately
                    # For WebSockets, files should be sent as base64 encoded strings in a separate message
                    gpt_id = await create_new_gpt(gpt)
                    logger.info(f"GPT created with ID: {gpt_id}")

                    # For file handling, the client would need to send files separately
                    response = {"message": "GPT created successfully!", "gpt_id": gpt_id}
                    await socket_manager.send_json(response, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except DuplicateKeyError as de:
                logger.error(f"DuplicateKeyError while creating GPT: {de}")
                await socket_manager.send_json({"error": "GPT name already exists."}, websocket)
            except HTTPException as he:
                logger.error(f"Error Code: {he}", exc_info=True)
                await socket_manager.send_json({"error": he.detail}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting GPTs
@router.websocket("/ws/get_gpts/{user_id}")
async def ws_get_gpts(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                logger.info(f"Logged User: {loggedUser}")

                if loggedUser != None and loggedUser != "N/A":
                    gpts = await get_gpts_for_user(loggedUser)

                    # If there is no gpts for a logged user. Create a gpt named Nia and update the usecases
                    if len(gpts) == 0:
                        gpt = GPTData(
                            name="gpt-4o",
                            description="Nia",
                            instructions="You are Nia, a virtual assistant. You can help the user with their tasks. You can answer questions, provide information, and help the user with their tasks. You can also ask follow-up questions to clarify the user's request.",
                            use_rag=True,
                            user=loggedUser,
                            use_case_id=""  # Set use_case_id to an empty string if not provided
                        )

                        print(f"Creating new GPT as no GPTs found for the user. {gpt}")
                        gpt_id = await create_new_gpt(gpt)
                        if gpt_id is not None:
                            logger.info(f"No GPTs found for the logged user. Created a new GPT with ID: {gpt_id}")
                            gpt._id = ObjectId(gpt_id)
                            gpts.append(gpt.model_dump(mode="json"))  # Convert GPTData to dict for JSON response

                        # update usecases
                        file_path = os.path.join(RAG_DOCUMENTS_FOLDER, "usecases_template.json")
                        try:
                            with open(file_path, "r", encoding='utf-8') as json_file:
                                usecases = json.load(json_file)
                                for usecase in usecases:
                                    usecase['gpt_id'] = ObjectId(gpt_id)
                                await update_usecases(gpt_id, usecases)
                                logger.info(f"usecases added to the gpt {gpt_id} successfully.")
                        except Exception as e:
                            logger.info(f"Error processing file: {str(e)}")
                            await socket_manager.send_json({"error": f"Error processing file: {str(e)}"}, websocket)

                    await socket_manager.send_json({"gpts": gpts}, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for chat
@router.websocket("/chat/{gpt_id}/{gpt_name}")
async def ws_chat(websocket: WebSocket, gpt_id: str, gpt_name: str, access_token: str = Query(...)):
    logger.info("Validation started")
    claims = await get_current_user_ws(access_token)
    if not claims:
        await websocket.close(code=4401)  # Custom code: Unauthorized
        logger.info("Unauthorized access attempt to WebSocket endpoint.")
        return
    
    await socket_manager.connect(websocket, gpt_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                user_message = data.get("user_message", "")
                if not user_message:
                    await socket_manager.send_json({"error": "Missing 'user_message' in request.", "type" : "error"}, websocket)
                    continue
                
                params = data.get("params", {})
                uploadedImage = data.get("uploadedImage")  # This would be base64 encoded image
                
                logger.info(f"Chat request received with GPT ID: {gpt_name} \n user message: {user_message} \n params: {params}")
                gpt = await get_gpt_by_id(gpt_id)
                await socket_manager.send_json({"response": f"Chat request received with GPT ID: {gpt_name} <br> user message: {user_message}"}, websocket)

                model_configuration = ModelConfiguration(**params)
                logger.info(f"Received GPT data: {gpt} \n Model Configuration: {model_configuration}")

                if gpt is None:
                    await socket_manager.send_json({"error": "GPT not found."}, websocket)
                    continue
                
                streaming_response = False
                response = await generate_response(streaming_response, user_message, model_configuration, gpt, uploadedImage, socket_manager, websocket)
                
                await socket_manager.send_json({
                    "response": response['model_response'], 
                    "total_tokens": response['total_tokens'] if response['total_tokens'] else 0, 
                    "follow_up_questions": response['follow_up_questions'],
                    "type": "chat_response"
                }, websocket)
            except HTTPException as he:
                logger.error(f"Error while getting response from Model. Details : \n {he.detail}", exc_info=True)
                await socket_manager.send_json({"error": f"Error while getting response from Model. Details : \n {he.detail}", "type" : "error"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e), "type" : "error"}, websocket)
    except WebSocketDisconnect as we:
        logger.error(f"Exception while socket handling {we.detail}", exc_info=True)
        socket_manager.disconnect(websocket, gpt_id)

# WebSocket endpoint for streaming chat
@router.websocket("/chat/stream/{gpt_id}/{gpt_name}")
async def ws_chat_stream(websocket: WebSocket, gpt_id: str, gpt_name: str, access_token: str = Query(...)):
    claims = await get_current_user_ws(access_token)
    if not claims:
        await websocket.close(code=4401)  # Custom code: Unauthorized
        logger.info("Unauthorized access attempt to WebSocket endpoint.")
        return
    
    await socket_manager.connect(websocket, gpt_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                user_message = data.get("user_message", "")
                if not user_message:
                    await socket_manager.send_json({"error": "Missing 'user_message' in request.", "type" : "error"}, websocket)
                    continue
                
                params = data.get("params", {})
                uploadedImage = data.get("uploadedImage")  # This would be base64 encoded image
                loggedUser = data.get("loggedUser")
                
                logger.info(f"Chat request received with GPT ID: {gpt_name} \n user message: {user_message}\n params: {params}")
                logger.info(f"Logged User: {loggedUser}")
                
                gpt = await get_gpt_by_id(gpt_id)

                model_configuration = ModelConfiguration(**params)
                logger.info(f"Received GPT data: {gpt} \n Model Configuration: {model_configuration}")

                if gpt is None:
                    await socket_manager.send_json({"error": "GPT not found."}, websocket)
                    continue
                
                streaming_response = True
                async for chunk in await generate_response(streaming_response, user_message, model_configuration, gpt, uploadedImage, socket_manager, websocket):
                    await socket_manager.send_json({"chunk": chunk, "type" : "stream_chunk"}, websocket) 
            except HTTPException as he:
                logger.error(f"Error while getting response from Model. Details : \n {he.detail}", exc_info=True)
                await socket_manager.send_json({"error": f"Error while getting response from Model. Details : \n {he.detail}", "type" : "error"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e), "type" : "error"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, gpt_id)

# WebSocket endpoint for updating instruction
@router.websocket("/ws/update_instruction/{gpt_id}/{gpt_name}/{usecase_id}/{user_id}")
async def ws_update_instruction(websocket: WebSocket, gpt_id: str, gpt_name: str, usecase_id: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                logger.info(f"Logged User: {loggedUser}")
                
                if loggedUser != None and loggedUser != "N/A":
                    result = await update_gpt_instruction(gpt_id, gpt_name, usecase_id, loggedUser)
                    logger.info(f"Instruction updated for GPT: {gpt_name}, result: {result}")

                    if result.modified_count == 1:
                        await socket_manager.send_json({"message": "Instruction updated successfully!", "gpt_name": gpt_name}, websocket)
                    elif result.modified_count == 0:
                        await socket_manager.send_json({"message": "No Changes in the instruction!", "gpt_name": gpt_name}, websocket)
                    else:
                        await socket_manager.send_json({"error": "GPT not found"}, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while updating instruction: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error Code: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for uploading document
@router.websocket("/ws/upload_document/{gpt_id}/{gpt_name}/{user_id}")
async def ws_upload_document_index(websocket: WebSocket, gpt_id: str, gpt_name: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                files_data = data.get("files", [])  # This would be base64 encoded files with metadata
                
                logger.info(f"Updating GPT with ID: {gpt_id} Name: {gpt_name}")
                gpts_collection = await get_collection("gpts")
                gpt = await gpts_collection.find_one({"_id": ObjectId(gpt_id)})
                logger.info(f"GPT Details: {gpt}")
                
                logger.info(f"Logged User: {loggedUser}")

                if loggedUser != None and loggedUser != "N/A":
                    if gpt is None:
                        await socket_manager.send_json({"error": "GPT not found"}, websocket)
                        continue
                        
                    gpt["user"] = loggedUser
                    gpt["use_case_id"] = gpt.get("use_case_id", "") 
                    gpt["use_rag"] = True
                
                    # Now you can access gpt as a dictionary
                    gpt = GPTData(**gpt)  # Validate and create GPTData instance
                    logger.info(f"Received GPT data: {gpt}")

                    # Convert base64 files to UploadFile objects
                    # This is a placeholder for the actual implementation
                    files = []  # This would need to be implemented

                    if files and len(files) > 0:
                        for file in files:
                            logger.info(f"Received files: {file.filename}")

                    logger.info(f"GPT : {gpt.name}, use_rag: {bool(gpt.use_rag)}")

                    file_upload_status = ""

                    if gpt.use_rag:
                        file_upload_status = await handle_upload_files(gpt_id, gpt, files)
                        logger.info(f"RAG Files uploaded successfully: {file_upload_status}")
                        await socket_manager.send_json({"message": "Document Uploaded Successfully!", "gpt_name": gpt_name, "file_upload_status": file_upload_status}, websocket)
                    else:
                        await socket_manager.send_json({"error": "GPT not found"}, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while updating GPT: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error Code: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for updating GPT
@router.websocket("/ws/update_gpt/{gpt_id}/{gpt_name}/{user_id}")
async def ws_modify_gpt(websocket: WebSocket, gpt_id: str, gpt_name: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                gpt_data = data.get("gpt", {})
                files_data = data.get("files", [])  # This would be base64 encoded files with metadata
                
                logger.info(f"Updating GPT with ID: {gpt_id} Name: {gpt_name}")
                logger.info(f"Logged User: {loggedUser}")
                
                if loggedUser != None and loggedUser != "N/A":
                    gpt_data["user"] = loggedUser
                    gpt_data["use_case_id"] = gpt_data.get("use_case_id", "") 
                
                    # Now you can access gpt as a dictionary
                    gpt = GPTData(**gpt_data)  # Validate and create GPTData instance
                    logger.info(f"Received GPT data: {gpt}")

                    # Convert base64 files to UploadFile objects
                    # This is a placeholder for the actual implementation
                    files = []  # This would need to be implemented

                    if files and len(files) > 0:
                        for file in files:
                            logger.info(f"Received files: {file.filename}")

                    result = await update_gpt(gpt_id, gpt_name, gpt)
                    logger.info(f"GPT : {gpt.name}, result: {result}, use_rag: {bool(gpt.use_rag)}")

                    file_upload_status = ""

                    if gpt.use_rag:
                        file_upload_status = await handle_upload_files(gpt_id, gpt, files)
                        logger.info(f"RAG Files uploaded successfully: {file_upload_status}")
                        
                    if result.modified_count == 1:
                        await socket_manager.send_json({"message": "GPT created successfully!", "gpt_name": gpt_name, "file_upload_status": file_upload_status}, websocket)
                    elif result.modified_count == 0:
                        await socket_manager.send_json({"message": "No Changes in the updated GPT!", "gpt_name": gpt_name, "file_upload_status": file_upload_status}, websocket)
                    else:
                        await socket_manager.send_json({"error": "GPT not found"}, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while updating GPT: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error Code: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for deleting GPT
@router.websocket("/ws/delete_gpt/{gpt_id}/{gpt_name}/{user_id}")
async def ws_remove_gpt(websocket: WebSocket, gpt_id: str, gpt_name: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                logger.info(f"Deleting GPT: {gpt_id} Name: {gpt_name}")
                logger.info(f"Logged User: {loggedUser}")

                # Delete the GPT
                gpt_delete_result = await delete_gpt(gpt_id, gpt_name)

                if gpt_delete_result.deleted_count == 1:
                    await socket_manager.send_json({"message": "GPT and Chat history removed successfully.!"}, websocket)
                else:
                    await socket_manager.send_json({"error": "GPT not found"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for deleting all GPTs
@router.websocket("/ws/delete_all_gpts/{user_id}")
async def ws_delete_all_gpts(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                logger.info(f"Logged User: {loggedUser}")
                
                if loggedUser != None and loggedUser != "N/A":
                    result = await delete_gpts(loggedUser)  # Delete all documents in the collection
                    if result.deleted_count > 0:
                        await socket_manager.send_json({"message": "All GPTs deleted successfully!"}, websocket)
                    else:
                        await socket_manager.send_json({"error": "No GPTs found"}, websocket)
                else:
                    await socket_manager.send_json({"error": "Unauthorized user"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting chat history
@router.websocket("/ws/chat_history/{gpt_id}/{gpt_name}/{user_id}")
async def ws_get_chat_history(websocket: WebSocket, gpt_id: str, gpt_name: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                logger.info(f"Fetching chat history for GPT: {gpt_id} Name: {gpt_name}")

                chat_history = await fetch_chat_history(gpt_id, gpt_name, max_tokens_in_conversation)

                # After saving the image, read its contents and encode the image as base64
                # The image URL will be saved in the chat. Use the URL to pick the image from the server
                for chat in chat_history:
                    if "chatimages" in chat["content"]:
                        uploads_directory = os.path.dirname(__file__)
                        imagePath = os.path.join(uploads_directory, chat["content"])
                        logger.info(f"Image URL found in chat history {imagePath}")

                if chat_history is None or chat_history == []:
                    await socket_manager.send_json({"error": "No Chats in the GPT"}, websocket)
                else:
                    # Reverse the list for linear view or to see proper conversation flow
                    await socket_manager.send_json({"chat_history": chat_history[::-1], "token_count": len(chat_history)}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting chat history for use case
@router.websocket("/ws/chat_history/{gpt_id}/{gpt_name}/{use_case_id}/{user_id}")
async def ws_get_chat_history_for_use_case(websocket: WebSocket, gpt_id: str, gpt_name: str, use_case_id: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                logger.info(f"Fetching chat history for GPT: {gpt_id} Name: {gpt_name}")

                if use_case_id == "all":
                    chat_history = await fetch_chat_history(gpt_id, gpt_name, max_tokens_in_conversation)
                else:
                    chat_history = await fetch_chat_history_for_use_case(use_case_id, gpt_id, gpt_name, max_tokens_in_conversation)
                logger.info(f"Chat history {chat_history}")

                # After saving the image, read its contents and encode the image as base64
                # The image URL will be saved in the chat. Use the URL to pick the image from the server
                for chat in chat_history:
                    if "chatimages" in chat["content"]:
                        uploads_directory = os.path.dirname(__file__)
                        imagePath = os.path.join(uploads_directory, chat["content"])
                        logger.info(f"Image URL found in chat history {imagePath}")

                if chat_history is None or chat_history == []:
                    await socket_manager.send_json({"error": "No Chats in the GPT"}, websocket)
                else:
                    # Reverse the list for linear view or to see proper conversation flow
                    await socket_manager.send_json({"chat_history": chat_history[::-1], "token_count": len(chat_history)}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for clearing chat history
@router.websocket("/ws/clear_chat_history/{gpt_id}/{gpt_name}/{user_id}")
async def ws_clear_chat_history(websocket: WebSocket, gpt_id: str, gpt_name: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                logger.info(f"Clearing chat history for GPT: {gpt_id} Name: {gpt_name}")

                result = await delete_chat_history(gpt_id, gpt_name)

                logger.info(f"Modified count: {result.modified_count}")
                
                if result.modified_count > 0:
                    await socket_manager.send_json({"message": "Cleared conversations successfully!"}, websocket)
                else:
                    await socket_manager.send_json({"error": "No messages found in GPT"}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting usecases
@router.websocket("/ws/usecases/{gpt_id}/{user_id}")
async def ws_fetch_usecases(websocket: WebSocket, gpt_id: str, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                result = await get_usecases(gpt_id)
                logger.info(f"Use cases fetched successfully: {len(result)}")
                await socket_manager.send_json({"message": "SUCCESS", "usecases": result}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while fetching usecases: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error occurred while fetching usecases: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting prompts
@router.websocket("/ws/get_prompts/{gpt_id}/{usecase}/{user}/{logged_user_id}")
async def ws_get_prompts_for_usecase(websocket: WebSocket, gpt_id: str, usecase: str, user: str, logged_user_id: str):
    await socket_manager.connect(websocket, logged_user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                
                # Fetch the use case details for the given GPT ID and use case ID
                prompts_default = await get_prompts(gpt_id, usecase, "Default")
                logger.info(f"Prompts Default: {prompts_default}")
                prompts = await get_prompts(gpt_id, usecase, user)
                
                # Merge prompts_default and prompts, avoiding duplicates by 'key'
                if prompts_default and prompts:
                    existing_keys = {p.get("key") for p in prompts}
                    merged_prompts = prompts.copy()
                    for p in prompts_default:
                        if p.get("key") not in existing_keys:
                            merged_prompts.append(p)
                    prompts = merged_prompts
                elif prompts_default:
                    prompts = prompts_default
                    
                logger.info(f"Prompts: {prompts}")

                if not usecase:
                    await socket_manager.send_json({"error": "Use case not found"}, websocket)
                else:
                    await socket_manager.send_json({"prompts": prompts}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while fetching prompts: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error occurred while fetching prompts: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, logged_user_id)

# WebSocket endpoint for updating prompt
@router.websocket("/ws/update_prompt/{gpt_id}/{usecase}/{user}/{prompt_title}/{logged_user_id}")
async def ws_update_prompt_for_usecase(websocket: WebSocket, gpt_id: str, usecase: str, user: str, prompt_title: str, logged_user_id: str):
    await socket_manager.connect(websocket, logged_user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                refinedPrompt = data.get("refinedPrompt", "")
                
                logger.info(f"Logged User: {loggedUser}")

                if not all([gpt_id, usecase, user, refinedPrompt]):
                    await socket_manager.send_json({"success": False, "error": "Missing required fields"}, websocket)
                    continue

                result = await update_prompt(gpt_id, usecase, user, refinedPrompt, prompt_title)
                logger.info(f"Prompt updated successfully: {result}")
                await socket_manager.send_json({"success": True}, websocket)
            except Exception as e:
                logger.error(f"Error in update_prompt_for_usecase: {e}", exc_info=True)
                await socket_manager.send_json({"success": False, "error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, logged_user_id)

# WebSocket endpoint for deleting prompt
@router.websocket("/ws/delete_prompt/{gpt_id}/{usecase}/{user}/{key}/{logged_user_id}")
async def ws_delete_prompt_for_usecase(websocket: WebSocket, gpt_id: str, usecase: str, user: str, key: str, logged_user_id: str):
    await socket_manager.connect(websocket, logged_user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                logger.info(f"Logged User: {loggedUser}")
            
                if not all([gpt_id, usecase, user, key]):
                    await socket_manager.send_json({"success": False, "error": "Missing required fields"}, websocket)
                    continue

                result = await delete_prompt(gpt_id, usecase, user, key)
                logger.info(f"Prompt deleted successfully: {result}")
                await socket_manager.send_json({"success": True}, websocket)
            except Exception as e:
                logger.error(f"Error in delete_prompt_for_usecase: {e}", exc_info=True)
                await socket_manager.send_json({"success": False, "error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, logged_user_id)

# WebSocket endpoint for getting logs
@router.websocket("/ws/logs/{user_id}")
async def ws_get_logs(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                log_file_path = "logs/app.log"  # Update with your actual log file path

                if not os.path.exists(log_file_path):
                    await socket_manager.send_json({"error": "Log file not found"}, websocket)
                    continue

                with open(log_file_path, "r") as f:
                    log_content = f.read()

                await socket_manager.send_json({"log_content": log_content}, websocket)
            except Exception as e:
                logger.error(f"Error: {str(e)}", exc_info=True)
                await socket_manager.send_json({"error": str(e)}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for getting deployed models
@router.websocket("/ws/deployedModels/{user_id}")
async def ws_getDeployedModelsFromAzure(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()  # Just to trigger the response, no data needed
            
            try:
                deployments = await getDeployments2()
                logger.info(f"Deployments fetched successfully: {len(deployments)}")
                if deployments is None:
                    await socket_manager.send_json({"message": "No deployments found"}, websocket)
                else:
                    await socket_manager.send_json({"message": "SUCCESS", "model_deployments": deployments}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while fetching deployments: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error occurred while fetching deployments: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

# WebSocket endpoint for refining prompt
@router.websocket("/ws/refinePrompt/{gpt_id}/{usecase}/{user}/{logged_user_id}")
async def ws_refinePrompt(websocket: WebSocket, gpt_id: str, usecase: str, user: str, logged_user_id: str):
    await socket_manager.connect(websocket, logged_user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                loggedUser = data.get("loggedUser")
                input_prompt = data.get("prompt", "")
                
                logger.info(f"Logged User: {loggedUser}")
                logger.info(f"Input prompt (Original) : {input_prompt} Length : {len(input_prompt)}")

                validator = PromptValidator()
                system_prompt = None

                if gpt_id is not None:
                    gpt_data = await get_gpt_by_id(gpt_id)
                    system_prompt = gpt_data["instructions"]

                # Process prompt
                response = await validator.process_prompt_optimized(input_prompt, system_prompt)
                refinedPrompt = response.refined_prompt
                promptTitle = response.title if hasattr(response, "title") else "Simple Prompt"
                logger.info(f"Title: {promptTitle} Refined prompt : {refinedPrompt} Length : {len(refinedPrompt)} ")
                update_response = await update_prompt(gpt_id, usecase, user, refinedPrompt, promptTitle)
                logger.info(f"Prompt updated: {update_response}")

                await socket_manager.send_json({"refined_prompt": response.dict() if hasattr(response, "dict") else response.__dict__}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while refining prompt: {e}", exc_info=True)
                await socket_manager.send_json({"refined_prompt": input_prompt}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, logged_user_id)

# WebSocket endpoint for getting image
@router.websocket("/ws/get_image/{user_id}")
async def ws_get_image(websocket: WebSocket, user_id: str):
    await socket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                image_path = data.get("imagePath", "")
                logger.info("Image path : " + image_path)
                
                image_path = urllib.parse.unquote(image_path)
                image_path = Path(image_path)

                if not image_path.is_file():
                    logger.info(f"Image not found in path: {image_path}")
                    await socket_manager.send_json({"error": "Image not found on the server"}, websocket)
                    continue
                
                logger.info(f"Fetching image from path: {image_path}")

                fullPath = os.path.join(os.path.dirname(__file__), image_path)
                logger.info(f"Full path of image : {fullPath}")

                with open(fullPath, "rb") as img_file:
                    logger.info("Reading image bytes")
                    image_bytes = img_file.read()
                    logger.info(f"Length of image : {len(image_bytes)}")

                    # Convert bytes to base64 string
                    base64_string = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Create the data URI
                    data_uri = f"data:image/jpeg;base64,{base64_string}"

                    await socket_manager.send_json({"image": data_uri}, websocket)
            except Exception as e:
                logger.error(f"Error occurred while fetching image: {e}", exc_info=True)
                await socket_manager.send_json({"error": f"Error occurred while fetching image: {e}"}, websocket)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket, user_id)

async def getDeployments2():
    deployed_model_names = []

    # Replace with your subscription ID, client ID, client secret, and tenant ID
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP_NAME")
    openai_account = os.getenv("OPENAI_ACCOUNT_NAME")
    
    # Your MSAL app credentials (Client ID, Client Secret, Tenant ID)
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET_VALUE")
    tenant_id = os.getenv("TENANT_ID")
    
    try:
        # Use the token with Azure SDK's client
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        
        # Create Cognitive Services management client with the token
        client = CognitiveServicesManagementClient(credential, subscription_id)
        
        logger.info("Starting to fetch deployments...")

        # Get all deployments in the subscription
        deployments = client.deployments.list(resource_group_name=resource_group, account_name=openai_account)

        if not deployments:
            logger.warning("No deployments found.")
        else:
            for deployment in deployments:
                logger.info(f"Deployment Name: {deployment.name}")
                deployed_model_names.append(deployment.name)

    except AzureError as e:
        logger.error(f"AzureError occurred: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")

    return deployed_model_names
