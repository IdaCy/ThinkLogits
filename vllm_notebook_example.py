#%%
# Install necessary libraries (run this cell once)
# You might need to restart the kernel after installation
!pip install vllm requests

#%% [markdown]
# ## Start the vLLM Server
#
# It's recommended to run the vLLM server in a **separate terminal window** rather than directly in the notebook/script, as it needs to run continuously in the background.
#
# Open a terminal and run:
# ```bash
# vllm serve "deepseek-ai/DeepSeek-R1-Zero"
# ```
# 
# Wait for the server to start up and indicate it's ready before running the next cell.

#%%
# Make the API call using the requests library

import requests
import json
import os

# Ensure the server is running before executing this cell!

# Define the API endpoint
# Use localhost if running locally, or the appropriate IP/hostname if running elsewhere
api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000")
url = f"{api_base}/v1/chat/completions"

# Define headers and payload
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-ai/DeepSeek-R1-Zero",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7, # Optional: Add generation parameters
    "max_tokens": 50    # Optional: Limit response length
}

# Send the POST request
try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    # Print the response
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
    print("Please ensure the vLLM server is running and accessible at the specified URL.")

except json.JSONDecodeError:
    print(f"Status Code: {response.status_code}")
    print("Could not decode JSON response:")
    print(response.text) 
# %%
