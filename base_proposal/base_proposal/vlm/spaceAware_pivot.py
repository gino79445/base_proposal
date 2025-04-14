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
                                    - yellow part of the map: Approximately the target object region.
                                    - red arrow: the direction between the current base position and the candidate base position.
                                    - Several numbered white circles: candidate base positions for you to move to.

                                Your task:
                                Given the instruction: "{INSTRUCTION}", choose {K} that best allow you to:
                                - Clearly observe the **key part of the object required for the task** (e.g., the mug handle).
                                - **Face the object from the front**, with a good viewing angle and manipulation feasibility.

                                Use the RGB image to understand the affordance and semantics of the target object.
                                Use the 2D map to reason about spatial layout and visibility.
                                At the end, directly return your answer as a JSON in the following format:
                                {{ "points": [] }}

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


## determine_base(path, part_to_grab_str, count_list)
# result = determine_base(
#    "./data/rgb.png",
#    "./data/cropped_occupancy_map.png",
#    "the handle of the mug",
#    [8, 9, 10, 11, 12, 13],
# )
# print(result)
