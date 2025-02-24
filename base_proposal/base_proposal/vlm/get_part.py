import json
import base64
import openai

from dotenv import load_dotenv
import os
load_dotenv()  # 讀取 .env 檔案
API_KEY  = os.getenv("API_KEY")  # 獲取 API Key
# Constants
MODEL = "gpt-4-turbo"

# Set the API key for OpenAI
openai.api_key = API_KEY

def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_chat_completion(model, messages, temperature=0.0):
    """Creates a chat completion request to OpenAI."""
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)


def determine_part_to_grab(image_path, instruction):
    """Determines which part to grab when picking up the target object."""

    base64_image = encode_image_to_base64(image_path)
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a robot capable of manipulating objects and moving."},
            {"role": "user", "content": [
                {"type": "text", "text": f"""If you want to {instruction}, which part of the object within the red box should you manipulate? 
                 you should make sure to output the component contained within the object in the red box.
                 Please must only output the result as follows: ["part","part",...]"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

    )
    # If you want manipulate the target in red box, which parts should you manipulate?
  #  print(response)
    json_response = json.loads(response.choices[0].message["content"])
    return json_response

#result = determine_part_to_grab("./data/original.png", "take out the baked  bread")
#print(result)
#for part in result:
#    print(part)

