import json
import base64
import openai

from dotenv import load_dotenv
import os
load_dotenv()  # 讀取 .env 檔案
API_KEY  = os.getenv("API_KEY")  # 獲取 API Key
import re
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

def get_point(image_path, INSTRUCTION, K):
    """Identifies the object in the image using OpenAI's model."""
    base64_image = encode_image_to_base64(image_path)

    # API call to identify the object in the image
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"""I am a wheeled robot that cannot go over objects.  This is the image I’m seeing right
                                            now.  I have annotated it with numbered circles.  Each number represent a general
                                            direction I can follow.  Now you are a five-time world-champion navigation agent and
                                            your task is to tell me which circle I should pick for the task of:  {INSTRUCTION}?
                                            Choose {K} best candidate numbers.  Do NOT choose routes that goes through objects.
                                            Skip analysis and provide your answer at the end in a json file of this form:
                                            {{"points":  [] }}"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]
    )
    
    # Parse the response to get the object name
    response_text = response.choices[0].message["content"].strip()  # 去除首尾空格與換行
    cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
    target_object = json.loads(cleaned_text)["points"]
    return target_object


#result = determine_affordance('./data/clustered_image.png',"keyhole and lock", "take out the baked bread", [1,2,3,4,5,6,7,8,9,10])
#result = identify_object_in_image('./data/annotated_image.png',"Take out the cookies from the cabinet.", 3)
