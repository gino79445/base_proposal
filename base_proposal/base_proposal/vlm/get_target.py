import json
import base64
import openai

from dotenv import load_dotenv
import os
load_dotenv()  # 讀取 .env 檔案
API_KEY  = os.getenv("API_KEY")  # 獲取 API Key
# Constants
MODEL = "gpt-4o"

# Set the API key for OpenAI
openai.api_key = API_KEY

def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_chat_completion(model, messages, temperature=0.0):
    """Creates a chat completion request to OpenAI."""
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)

def identify_object_in_image(image_path):
    """Identifies the object in the image using OpenAI's model."""
    base64_image = encode_image_to_base64(image_path)

    # API call to identify the object in the image
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is beneath the red pyramid? Provide only the name of the object. If the object is found beneath the red pyramid, output the result as follows: {\"name\": \"object\"}. Otherwise, output the result as follows: {\"name\": \"No\"}."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]
    )
    
    # Parse the response to get the object name
    target_object = json.loads(response.choices[0].message["content"])["name"]
    return target_object

