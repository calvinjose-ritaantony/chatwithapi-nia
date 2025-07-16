import os
import json
import asyncio
import logging
from datetime import datetime
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from dotenv import load_dotenv
from ModelResponse import ModelResponse

load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)

# Initialize the Azure OpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint = os.getenv("GPT4O_ENDPOINT_URL"), 
    api_key=os.getenv("GPT4O_API_KEY"),  
    api_version=os.getenv("GPT4O_API_VERSION")
)

deployment_name = os.getenv("GPT4O_MODEL_NAME")

conversations = [
    {"role": "system", "content":
        """You are AI assistant specialized in forming structured outputs.

        You MUST return a structured response that includes both readable text and structured data for tables and/or graphs as appropriate.

        When returning information, structure your response according to these guidelines:
        1. Include a conversational text response in model_response
        2. If the data is tabular, set has_table to true and provide the table_data structure
        3. If a visualization would be helpful, set has_graph to true and provide appropriate graph_data
        4. Use reasoning field to explain your approach (this won't be shown to the user)
        5. Suggest follow-up questions in the follow_up_questions array

        FORMAT YOUR RESPONSE:
        - Address the user's question directly
        - For tables: Include headers and rows with customer orders
        - For graphs: Use appropriate chart type to visualize order status distribution
        - Use appropriate types for fields (dates as datetime, numbers as floats)
        - Be precise about dates and statuses, as users rely on this information for planning
        
        Remember to structure your response according to the provided ModelResponse schema."""},
    {"role": "user", "content": "Track the orders of Chris Miller and show me a summary chart of order status"}
]

async def main():
    try:
        # Uncomment to use OpenAI API
        response: ParsedChatCompletion = await client.beta.chat.completions.parse(
            model=deployment_name,
            messages=conversations,
            temperature=0.3,
            top_p=0.95,
            seed=100,
            stop=None,
            response_format=ModelResponse
        )

        model_response: ModelResponse = response.choices[0].message.parsed
        print(model_response)
        
        # # Example of accessing specific parts
        # if model_response.has_table:
        #     print("\nTable Headers:", model_response.table_data[0].headers)
            
        # if model_response.has_graph:
        #     print("\nGraph Type:", model_response.graph_data[0].chart_type)
            
        # # Example of using the data to render in a UI
        # if model_response.has_table:
        #     print("\nTable for UI rendering:")
        #     headers = model_response.table_data[0].headers
        #     print(" | ".join(headers))
        #     print("-" * (len(" | ".join(headers))))
        #     for row in model_response.table_data[0].rows:
        #         print(" | ".join([str(cell) for cell in row]))

    except Exception as final_ex:
        logger.error(f"Exception occurred: {final_ex}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())