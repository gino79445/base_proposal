import json
import base64
import openai

from dotenv import load_dotenv
import os
import re

load_dotenv()  # 讀取 .env 檔案
API_KEY = os.getenv("API_KEY")  # 獲取 API Key
# Constants
MODEL = "gpt-4o"

# Set the API key for OpenAI
openai.api_key = API_KEY


def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_chat_completion(model, messages, temperature=0.5):
    """Creates a chat completion request to OpenAI."""
    return openai.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )


def get_point(image_path1, image_path2, INSTRUCTION, K):
    """Determines which part to grab when picking up the target object."""
    #  and there is a red point mrked on the {part} of the target object.
    #
    #            Step 1: Understand the target and its position
    #            Identify the {part} of the target object in the red frame and note its location in the image relative to the grid and candidate points.

    base64_image1 = encode_image_to_base64(image_path1)
    base64_image2 = encode_image_to_base64(image_path2)
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a robot capable of manipulating objects and moving.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                                You are a professional mobile robot agent.
                                You are given two images:
                                1. The first image is an RGB image showing the current scene from your onboard camera. 

                                2. The second image is a top-down 2D map. The map contains:
                                    - A blue circle: your current base position.
                                    - black part of the map: the free space.
                                    - white part of the map: the occupied space.
                                    - Several numbered white circles with a blue outline labeled IDs(0~14): candidate base positions (obstacle-free) for you to move to.
                                
                                Important:
                                    You must use both the onboard RGB image and the top-down map together.

                                Your task:
                                Given the instruction: "{INSTRUCTION}", choose {K} that best allow you to:
                                - Please determine which key part (e.g., the mug handle) of the object should be manipulated
                                - Clearly observe the **key part of the object required for the task** 
                                   when you are in the candidate base position and facing the target. 
                                - Easily finish the task when you are in the candidate base position and facing the target.

                                At the end, directly return your answer as a JSON in the following format: {{ "points": [] }}
                                Only return the JSON. Do not include explanation or reasoning.
                                """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image1}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image2}"},
                    },
                ],
            },
        ],
    )
    response_text = response.choices[0].message.content
    cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
    target_object = json.loads(cleaned_text)["points"]
    return target_object
