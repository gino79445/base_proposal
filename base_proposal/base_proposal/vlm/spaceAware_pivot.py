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
                        You are now a wheeled robot. 
                        The first image I’m seeing right now.
                        The second image is a bird’s eye view, where several candidate base positions are marked with numbered labels. 
                        The red area represents your current base position.The green area indicates the location of your target object.
                        You have annotated it with numbered circles.  Each number represent a general
                                            direction you can follow.  Now you are a five-time world-champion navigation agent and
                                            your task is to tell me which circle I should pick for the task of:  {INSTRUCTION}?
                                            Please analyze the semantics meaning with the rgb image and the instruction and understand the spatial relationship between the object and the candidate points with the bird's eye view image.
                                            Choose {K} best candidate numbers.
                                            Skip analysis and provide your answer at the end in a json file of this form:
                                            {{"points":  [] }}

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
    print(response_text)
    cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
    target_object = json.loads(cleaned_text)["points"]
    print(target_object)
    return target_object


## determine_base(path, part_to_grab_str, count_list)
# result = determine_base(
#    "./data/rgb.png",
#    "./data/cropped_occupancy_map.png",
#    "the handle of the mug",
#    [8, 9, 10, 11, 12, 13],
# )
# print(result)
