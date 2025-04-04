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


def parse_instruction(instruction):
    """Determines which part to grab when picking up the target object."""
    #  and there is a red point mrked on the {part} of the target object.
    #
    #            Step 1: Understand the target and its position
    #            Identify the {part} of the target object in the red frame and note its location in the image relative to the grid and candidate points.

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
                        You are a robot using your left hand and not holding anything. The instruction '{instruction}'  is what you need to follow. Please parse the instruction into two parts: 
                        the navigation destination and the manipulation action.
                        For example, for the instruction "Throw the garbage on the cabinet into the trash bin," the parsed parts would be:
                        ["the garbage on the cabinet", "pick up the garbage", "trash bin", "throw the garbage"].

                        Please output the result in the following format:

                         ["navigation destination","manipulation",...,"navigation destination","manipulation"]
                                                
                        """,
                    },
                ],
            },
        ],
    )
    json_response = json.loads(response.choices[0].message.content)
    return json_response


a = parse_instruction("put the red mug on the shelf")
print(a)
## determine_base(path, part_to_grab_str, count_list)
# result = determine_base(
#    "./data/rgb.png",
#    "./data/cropped_occupancy_map.png",
#    "the handle of the mug",
#    [8, 9, 10, 11, 12, 13],
# )
# print(result)
