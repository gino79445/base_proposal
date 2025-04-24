import json
import base64
import openai
from dotenv import load_dotenv
import os
import re

load_dotenv()  # 讀取 .env 檔案
API_KEY = os.getenv("API_KEY")  # 獲取 API Key
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


def determine_affordance(image_path, instruction, number_list):
    """Determines which part to grab when picking up the target object."""

    #              {"type": "text", "text": f"""This image was taken by the camera on you.
    #                                           The object in the red bounding box is your target object for operation.
    #                                           The target object has been marked with colored blocks, and each block is labeled with a red ID number.
    #                                           Based on this image, Based on this image, if you want to manipulate the {part},
    #                                           please select an area that contains the {part} of the target object.
    #                                           In the image, some candidate areas
    #                                           on the target object have been labeled with IDs. Please select a single candidate red point
    #                                           that meets the requirements above and provide the selected red point's ID.
    #                                           Please make sure the point's ID on the image and must output the result as follows: {{"id":id}}."""},
    #              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    base64_image = encode_image_to_base64(image_path)
    # list to string
    number_list = str(number_list)
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

                            The object with labeled points in this image is your target object.
                            You are asked to select one candidate point that you should touch or face when performing the following instruction:
                            "{instruction}"
                            Please evaluate the points based on their affordance — how suitable each point is for performing the intended action.
                            Candidate points are labeled with IDs: {number_list}.
                            Return only one point ID that best satisfies the condition.
                            Please make sure the point's ID on the image and must output the result as follows: {{"id":id}}.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
    )
    response_text = response.choices[0].message.content
    cleaned_text = re.sub(r"```json\n?|```", "", response_text).strip()
    target_object = json.loads(cleaned_text)["id"]
    return target_object


# result = determine_base('./data/clustered_image.png', 'take out the cookie')

# result = determine_affordance('./data/clustered_image.png',"keyhole and lock", "take out the baked bread", [1,2,3,4,5,6,7,8,9,10])
#
# print(result)
