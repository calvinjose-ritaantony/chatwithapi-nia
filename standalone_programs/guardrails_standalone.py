import sys
import os
import asyncio
import huggingface_hub
# Add the src directory to sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, CompetitorCheck
from dependencies import NiaAzureOpenAIClient


from dotenv import load_dotenv # For environment variables (recommended)

load_dotenv()  # Load environment variables from .env file

AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_MODEL_API_VERSION = os.getenv("API_VERSION")

# Login to Hugging Face Hub using token from environment variable
huggingface_hub.login(token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))

# 1. Configure OpenAI
GPT_MODEL = "gpt-4o"

# Initialize Azure OpenAI client using NiaAzureOpenAIClient
nia_azure_client = NiaAzureOpenAIClient(
    AZURE_ENDPOINT_URL,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_MODEL_API_VERSION,
    False
)
openai_client = nia_azure_client.get_azure_client()

# 2. Build guards
input_guard = Guard().use(
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION)  # reject toxic user prompts
)

output_guard = Guard().use_many(
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.REASK),      # sanitize toxic LLM outputs
    CompetitorCheck(["Apple", "Microsoft", "Google"], on_fail=OnFailAction.EXCEPTION)
)

async def chat():
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            break

        # Input guard
        try:
            input_guard.validate(user)
        except Exception as e:
            print(f"[Input rejected] {e}")
            continue

        # Call LLM
        resp = await openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role":"user","content":user}]
        )
        llm_out = resp.choices[0].message.content

        print("\n[Raw LLM output]")
        print(llm_out)

        # Output guard
        try:
            output_guard.validate(llm_out)
            print("\n[Final Response]")
            print(llm_out)
        except Exception as e:
            print(f"\n[Output guard failed] {e}")

if __name__ == "__main__":
    asyncio.run(chat())
