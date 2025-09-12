import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from openai import AsyncAzureOpenAI
from ConnectionManager import ConnectionManager

load_dotenv()  # Load environment variables from .env file

APP_SCOPE = os.getenv("APP_SCOPE")
AUTHORITY= os.getenv("AUTHORITY")

# Application (client) ID of app registration
CLIENT_ID = os.getenv("CLIENT_ID")
# Application's generated client secret: never check this into source control!
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")

# Azure Blob Storage - Used for storing image uploads
UPLOAD_FOLDER = "uploads"
AZURE_BLOB_STORAGE_CONNECTION_URL=os.getenv("BLOB_STORAGE_CONNECTION_STRING")
AZURE_BLOB_STORAGE_CONTAINER=os.getenv("BLOB_STORAGE_CONTAINER_NAME")
AZURE_BLOB_STORAGE_ACCOUNT_NAME=os.getenv("BLOB_STORAGE_ACCOUNT_NAME")
AZURE_BLOB_STORAGE_ACCESS_KEY=os.getenv("BLOB_STORAGE_ACCESS_KEY")
RAG_DOCUMENTS_FOLDER = os.path.join(UPLOAD_FOLDER, "ragdocuments")
 
REDIRECT_PATH = "/getAToken"  # Used for forming an absolute URL to your redirect URI.

ENDPOINT = 'https://graph.microsoft.com/v1.0/me'  
SCOPE = ["User.Read"]

# Google Custom Search API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GOOGLE_SEARCH_ENGINE_URL = os.getenv("GOOGLE_SEARCH_ENGINE_URL")
 

# Sonar Perplexity API Configuration for Realtime web search
SONAR_PERPLEXITY_API_KEY = os.getenv("SONAR_PERPLEXITY_API_KEY")
SONAR_PERPLEXITY_URL = os.getenv("SONAR_PERPLEXITY_URL")
SONAR_PERPLEXITY_MODEL = os.getenv("SONAR_PERPLEXITY_MODEL", "sonar-pro")  # Default to sonar-pro if not se

# Azure GPT 4o parameters
GPT_4o_MODEL_NAME = os.getenv("GPT4O_MODEL_NAME")
GPT4O_2_MODEL_NAME = os.getenv("GPT4O_2_MODEL_NAME")
GPT_4o_API_KEY=os.getenv("GPT4O_API_KEY")
GPT_4o_ENDPOINT_URL=os.getenv("GPT4O_ENDPOINT_URL")
GPT_4o_API_VERSION = os.getenv("GPT4O_API_VERSION")

AZURE_ENDPOINT_URL_2 = os.getenv("AZURE_ENDPOINT_URL_2")
OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_2")
API_VERSION_2 =os.getenv("API_VERSION_2")

 # Azure Open AI Clients for different tasks
azure_openai_client =  AsyncAzureOpenAI(
        azure_endpoint=GPT_4o_ENDPOINT_URL,
        api_key=GPT_4o_API_KEY,
        api_version=GPT_4o_API_VERSION)


blob_service_client = BlobServiceClient(f"https://{AZURE_BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=AZURE_BLOB_STORAGE_ACCESS_KEY
)


# Tells the Flask-session extension to store sessions in the filesystem
SESSION_TYPE = "filesystem"


# Create connection manager
socket_manager = ConnectionManager()


