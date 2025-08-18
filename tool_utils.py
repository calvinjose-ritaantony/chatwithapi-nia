
from data.NiaTool import NiaTool
from mongo_service import get_usecases_list

async def azure_ai_search_tool(gpt_id: str, use_case: str) -> NiaTool:
    return NiaTool(
        tool_name="get_data_from_azure_search", 
        tool_description="Function that allows NIA to get data from Azure AI Search to get E-Commerce order related details",
        type="pre_response",
        tool_definition={
            "type": "function",
            "function": {
                "name": "get_data_from_azure_search",
                "description": "Fetch the e-commerce order related documents from Azure AI Search for the given user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The user query related to e-commerce orders, products, reviews, status, analytics etc, e.g. find all orders by Chris Miller, Summarize the reviews of product P015",
                        },
                        "use_case": {
                            "type": "string", 
                            "enum": await get_usecases_list(gpt_id),
                            "description": f"The actual use case of the user query, e.g. {use_case}"
                            },
                        "get_extra_data":{
                            "type": "boolean",
                            "description": "If true, fetch the extra data from NIA Finolex Search Index. If false, fetch the data from the use case index"
                        }
                    },
                    "required": ["search_query", "use_case", "get_extra_data"],
                },
            }
        }
    )

async def web_search_tool() -> NiaTool:
    return NiaTool(
        tool_name="get_data_from_web_search", 
        tool_description="Function that allows NIA to get the latest, real-time data via web search",
        type="pre_response",
        tool_definition={
            "type": "function",
            "function": {
                "name": "get_data_from_web_search",
                "description": "Fetch live public information from the web for the given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Topic to search on the web, e.g., 'latest Nifty 50 index value'"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region code for results, e.g., 'IN', 'US'",
                            "default": "IN"
                        }
                    },
                    "required": ["search_query"]
                }
            }
        }
    )

async def write_response_to_pdf_tool() -> NiaTool:
    return NiaTool(
        tool_name="write_response_to_pdf", 
        tool_description="Function that allows NIA to write the response to a pdf file",
        type="post_response",
        tool_definition={
                "type": "function",
                "function": {
                    "name": "write_response_to_pdf",
                    "description": "If the user requests to download the response as PDF, this function will generate the PDF from the response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response_text": {  
                                "type": "string",
                                "description": "The response text from the model"
                            },
                            "file_name": {  
                                "type": "string",
                                "description": "The file name for the PDF, e.g., 'ANALYZE_SPENDING_PATTERN_CHRIS_MILLER_RESPONSE'",
                            }
                        },
                        "required": ["response_text"],
                    },
                }
            }
    )

