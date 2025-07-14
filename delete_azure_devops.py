import os 
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

# This script deletes all work items in a specified Azure DevOps project.

# Replace with your Azure DevOps org, project, and PAT
ORGANIZATION = "COE-CX"
PROJECT = "NIA"
PAT = os.getenv("AZURE_DEV_OPS_PERSONAL_ACCESS_TOKEN")

# Base URL
BASE_URL = f"https://dev.azure.com/{ORGANIZATION}/{PROJECT}/_apis/wit/workitems"
API_VERSION = "7.0"

# Authentication
auth = HTTPBasicAuth('', PAT)

# Step 1: Get all work item IDs
print("Fetching all work item IDs...")

query_url = f"https://dev.azure.com/{ORGANIZATION}/{PROJECT}/_apis/wit/wiql?api-version={API_VERSION}"
query = {
    "query": f"SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = '{PROJECT}'"
}

resp = requests.post(query_url, json=query, auth=auth)
resp.raise_for_status()
ids = [item["id"] for item in resp.json()["workItems"]]

if not ids:
    print("No work items found.")
else:
    print(f"Found {len(ids)} work items. Starting deletion...")

# Step 2: Delete all work items
for wid in ids:
    delete_url = f"https://dev.azure.com/{ORGANIZATION}/_apis/wit/workitems/{wid}?api-version={API_VERSION}"
    del_resp = requests.delete(delete_url, auth=auth)
    if del_resp.status_code == 204:
        print(f"Deleted work item {wid}")
    else:
        print(f"Failed to delete {wid}: {del_resp.status_code} {del_resp.text}")
