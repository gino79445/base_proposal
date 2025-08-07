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
                               You are provided with two images:
                               1. RGB image from the onboard camera:
                                   - Black point outlined in orange labeled "A": marks a floor point likely indicating the affordance direction.
                                   - Colored lines on the ground: Outward vectors from the target object center; the line marked "A" is the affordance direction.
                               2. Top-down 2D map:
                                   - Blue circle: your current base position.
                                   - Black areas: free space.
                                   - White areas: occupied space.
                                   - Yellow area: approximate target object area (matching the RGB image).
                                   - Numbered white circles with blue outlines (IDs 0–19): candidate base positions (obstacle-free).
                                   - Black point outlined in orange "A": same affordance point as in the RGB image.
                                   - Orange arrow at "A": indicates the affordance direction on the map.
                                   - The orange region near the arrow at "A": indicates an area with good candidate base positions.
                                   - Colored lines on the ground: match the directions seen in the RGB image (each drawn every 30°).

                               Important:There is only one number for each numbered white circle with blue outlines on the top-down map.

                               Your task:
                                   Given the instruction "{INSTRUCTION}", select the {K} candidate base positions that best:
                                       - Allow you to clearly observe the key part of the object required for the task when positioned there, facing the target.
                                       - Align the robot to face the primary interaction side based on the affordance’s functional geometry.
                               Steps:
                                   1. Identify the key part of the target object that needs to be manipulated (e.g., mug handle).
                                   2. Align the RGB image and top-down map using the colored lines (same colors,same order, centered on the object, every 30°).
                                   3. Select the best candidate base positions:
                                       - The direction marked at point "A" roughly indicates the main reference direction
                                       and good base positions typically lie within the orange area on the top-down map.
                                       - If the orange area on the top-down map is visible, please select the candidate base positions 
                                       that are in the orange area on the top-down map and skip the next step.
                                       If the orange area on the top-down map is not visible, you should first use the RGB image to
                                       which direction line on the floor corresponds to the color that allows the robot to clearly see the key part
                                       Once the color is determined, use the top-down map to find the corresponding directional line.
                                       Good candidate points should be near the directional line.
                                   4. Return the selected candidate base positions and make sure the number is only one
                                   for each numbered white circle with blue outlines on the top-down map.
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
