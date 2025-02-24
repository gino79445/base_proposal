import json
import base64
import openai

from dotenv import load_dotenv
import os
load_dotenv()  # 讀取 .env 檔案
API_KEY  = os.getenv("API_KEY")  # 獲取 API Key
# Constants
API_KEY = "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"
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

def confirm_part_in_image(part, image):
    """Confirms the presence of a part in the image."""
    base64_image = encode_image_to_base64(image)
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a robot capable of grasping objects and moving"},
            {"role": "user", "content": [
                {"type": "text", "text": f'Is there {part} in the picture? Please output the result as follows: {{"answer": "Yes"}} or {{"answer": "No"}}'},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]
    )
    return json.loads(response.choices[0].message["content"])["answer"]

