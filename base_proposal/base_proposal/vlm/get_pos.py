import json
import base64
import openai

from dotenv import load_dotenv
import os

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


def determine_base(image_path1, image_path2, part, number_list):
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
                        You are now a mobile robot. Suppose you need to perform an operation on {part}.

                        The first image is your RGB perspective view.
                        The second image is a bird’s eye view, where several candidate base positions are marked with numbered labels. 
                        The green area indicates the location of the mug, and the blue area represents your current base position. 
                        The viewing direction from the blue position is the same as in the first image.
                        Given this information, which of the numbered candidate points should you move to in order to see the {part} 
                        and perform the operation on it ?
                        You need to move to a candidate position where are able to reach and manipulate the {part}.

                        In the image, candidate points are labeled with IDs {number_list}. 
                        Once the best candidate point is identified, must only output the result in the following format: {{"id": id}}.
                        
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
    json_response = json.loads(response.choices[0].message.content)
    json_response = json_response["id"]
    return json_response


## determine_base(path, part_to_grab_str, count_list)
# result = determine_base(
#    "./data/rgb.png",
#    "./data/cropped_occupancy_map.png",
#    "the handle of the mug",
#    [8, 9, 10, 11, 12, 13],
# )
# print(result)
