import os
import requests
import json
import logging
from dotenv import load_dotenv
#from bs4 import BeautifulSoup
from openai import RateLimitError

from app_config import SONAR_PERPLEXITY_API_KEY, SONAR_PERPLEXITY_MODEL, SONAR_PERPLEXITY_URL
# from utils.constants import ELEMENTS_TO_STRIP
# from utils.prompts import WEB_SEARCH_DATA_SUMMARIZATION_SYSTEM_PROMPT, WEB_SEARCH_KEYWORD_CONSTRUCTION_SYSTEM_PROMPT, WEB_SEARCH_KEYWORD_CONSTRUCTION_USER_PROMPT

# Load environment variables from .env file
load_dotenv()  

# Create a logger for this module
logger = logging.getLogger(__name__)

timeout = 20

async def search_web_with_sonar(query: str):

    citations = []
    grounding_data_from_web = None

    try:
        # Set up the API endpoint and headers
        headers = {
            "Authorization": f"Bearer {SONAR_PERPLEXITY_API_KEY}",  # Replace with your actual API key
            "Content-Type": "application/json"
        }

        # Define the request payload
        payload = {
            "model": SONAR_PERPLEXITY_MODEL,
            "messages": [ 
                {
                    "role" : "system",
                    "content" : """You are a helpful AI assistant.

                        Task:
                        1. Your task is to to real-time web search for finding latest information on any given topic.

                        Rules:
                        1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
                        2. Do not show the intermediate steps information.
                        

                        Steps:
                        1. Decide if the answer should be a brief sentence or a list of suggestions.
                        2. If it is a list of suggestions, first, write a brief and short, natural introduction based on the original query.
                        3. Followed by a list of suggestions, each suggestion should be split by two newlines."""
                },
                {
                    "role": "user", 
                    "content": f"""
                        Follow the below steps to search the web:
                        - Optimize the given query and extract SEO optimized web search key terms.
                        - Use the optimized keywords to search the web.

                        Query: {query}
                        """
                }
            ], #3. Please return the data as a JSON object in the format specified in the "response_format" section.
            # "response_format": {
            #     "type": "json_schema",
            #     "json_schema": {
            #         "schema": {
            #         "type": "object",
            #         "properties": {
            #             "topic": {"type": "string"},
            #             "summary": {"type": "string"},
            #             "citations": {
            #                 "type": "array",
            #                 "references": {"type": "string"}
            #             }
            #         },
            #         "required": ["topic", "summary", "citations"]
            #         }
            #     }
            # }
        }

        # Make the API call
        response = requests.post(SONAR_PERPLEXITY_URL, headers=headers, json=payload)

        logger.info(f"Response from SONAR : {response.json()}")

        if response is not None:
            citations = response.json().get("citations", [])
            grounding_data_from_web = response.json()["choices"][0]['message']['content']

        # Print the AI's response
        logger.info(f"Response from SONAR : \n {response.json()}") # replace with print(response.json()["choices"][0]['message']['content']) for just the content
    except Exception as e:
        logger.error(f"Error occurred while calling the Sonar Perplexity API: {e}", exc_info=True)

    return grounding_data_from_web, citations

# async def google_search(query: str, num_results: int = 3):
#     """Search Google Custom Search API and return the top URLs."""
    
#     url = GOOGLE_SEARCH_ENGINE_URL
#     params = {
#         "key": GOOGLE_API_KEY,
#         "cx": GOOGLE_CSE_ID,
#         "q": query,
#         "num": num_results,
#         #"siteSearch": "amazon.in, flikart.com"  # Limit search to Amazon India
#     }

#     try:
#         response = requests.get(url, params=params, timeout=timeout)
#         response.raise_for_status()
#         data = response.json()
#         return [item["link"] for item in data.get("items", [])]
#     except requests.RequestException as e:
#         logger.error(f"Error fetching search results: {e}")

#     return []

# # async def extract_text_from_url(url):
#     """Extract main content from a webpage using BeautifulSoup 4.12.2."""

#     headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
#     #headers = {"User-Agent": "Mozilla/5.0"}  # Prevent bot detection

#     try:
#         response = requests.get(url, headers=headers, timeout=timeout)
#         response.raise_for_status()

#         #logger.info(f"Google Search Results {response.text}")

#         soup = BeautifulSoup(response.content, "html5lib")  # Use html5lib parser for advanced parsing
#         #logger.info(f"Google Search Results (html5lib) : {soup.prettify()}")

#         # soup = BeautifulSoup(response.content, "lxml")  # Use lxml parser for faster parsing
#         # logger.info(f"Google Search Results  (lxml) : {soup.prettify()}")
        
#         # Remove unnecessary tags (script, style, etc.)
#         for script in soup(ELEMENTS_TO_STRIP):
#             script.decompose()
        
#         # Extract text from <p> tags
#         paragraphs = soup.find_all("p")
#         text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])

#         return text[:3000]  # Limit text to 3000 characters for API efficiency
#     except requests.RequestException as e:
#         logger.error(f"Error extracting content from {url}: {e}", exc_info=True)
        
#     return ""

# # async def search_web(query: str, deployment_name: str):
#     """Search Google, extract content, and generate summaries."""

#     web_search_conversations = []
#     urls = []
#     texts = []
#     web_search_summary = "No Data from web search"

#     logger.info(f"Original Query {query}")

#     try:
#         # Step 1: Understand the query and extract optimized keywords for web search
#         web_search_conversations.append({"role": "system", "content": WEB_SEARCH_KEYWORD_CONSTRUCTION_SYSTEM_PROMPT})
#         web_search_conversations.append({"role": "user", "content": WEB_SEARCH_KEYWORD_CONSTRUCTION_USER_PROMPT.format(query=query)})
        
#         response = await azure_openai_client.chat.completions.create(
#             model=deployment_name,
#             messages=web_search_conversations,
#             max_completion_tokens=20,
#             temperature=0.1 # deterministic to get most probable keywords
#         )

#         # Process the model's response
#         keywords_for_web_search = response.choices[0].message.content if len(response.choices) > 0 else query
#         logger.info(f"keywords_for_web_search {keywords_for_web_search}")
#         web_search_conversations.clear() # Clear the messages list because we do not need the system message, user message in this function

#         # Step 2: Peform google SEO search, get the top URLs and search the web
#         search_results = await google_search(query=keywords_for_web_search, num_results=3) # the query can also be sent to a llm for verifying the intent and generate a search string, then use function calling to do web search and fetch urls
#         logger.info(f"search_results {search_results}")
        
#         if search_results is not None or len(search_results) > 0:
#             for url in search_results:
#                 text = await extract_text_from_url(url)
#                 if text is not None and len(text) > 0:
#                     texts.append(text)
#                     urls.append(url)
            
#             logger.info(f"Texts from web search: {texts} and urls : {urls}")

#             # Step 3 : Summarize the web search results using a model
#             web_search_summary = await summarize_web_search_results(texts=texts, deployment_name=deployment_name)
#         else:
#             web_search_summary = "No Data from web search"

#     except RateLimitError as rle:
#         logger.error(f"RateLimitError occurred while calling the function: {rle}", exc_info=True)
#         web_search_summary = "No Data from web search" # this error message should get added to the final summary so just send no data when exception
#     except Exception as e:
#         logger.error(f"Error occurred while calling the function: {e}", exc_info=True)
#         web_search_summary = "No Data from web search"
#     finally:
#         web_search_conversations.clear() # Clear the messages list because we do not need the system message, user message in this function
#         logger.info(f"web_search_summary : {web_search_summary} \n urls : {urls}")

#     return web_search_summary, urls

# # async def summarize_web_search_results(texts, deployment_name):
#     """Search the web and generate summaries."""

#     web_search_summarizing_conversations = []
#     web_search_summary = "No data"

#     try:
#         if len(texts) > 0:
#             web_search_summarizing_conversations.append({"role": "system", "content": WEB_SEARCH_DATA_SUMMARIZATION_SYSTEM_PROMPT})
#             web_search_summarizing_conversations.append({"role": "user", "content": f"Web Search Results \n {json.dumps(texts)}"})
            
#             # First API call: Ask the model to use the function
#             response = await azure_openai_client.chat.completions.create(
#                 model=deployment_name,
#                 messages=web_search_summarizing_conversations,
#                 max_completion_tokens=200,
#                 temperature=0.3,
#                 frequency_penalty=0.75,
#                 presence_penalty=0.25
#             )

#             logger.info(f"Summary : {response}")

#             # Process the model's response
#             web_search_summary = response.choices[0].message.content if len(response.choices) > 0 else "No Data"
#     except RateLimitError as rle:
#         logger.error(f"RateLimitError occurred while calling the function: {rle}", exc_info=True)
#         web_search_summary = "No Data" # this error message should get added to the final summary so just send no data when exception
#     except Exception as e:
#         logger.error(f"Error occurred while calling the function: {e}", exc_info=True)
#         web_search_summary = "No Data"
#     finally:
#         web_search_summarizing_conversations.clear() # Clear the messages list because we do not need the system message, user message in this function
#         logger.info(f"web_search_summary {web_search_summary}")
    
#     return web_search_summary

# Example usage
# query = "Impact of AI on e-commerce"
# search_results = search_web(query)

# for url, text in search_results.items():
#     print(f"\n {url}\n Text:\n{text}\n")
