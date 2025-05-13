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


def create_chat_completion(model, messages, temperature=0.0):
    """Creates a chat completion request to OpenAI."""
    return openai.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )


def get_affordance_direction(image_path, INSTRUCTION, IDs):
    """Determines which part to grab when picking up the target object."""
    #  and there is a red point mrked on the {part} of the target object.
    #
    #            Step 1: Understand the target and its position
    #            Identify the {part} of the target object in the red frame and note its location in the image relative to the grid and candidate points.

    base64_image = encode_image_to_base64(image_path)
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
                                The image is a rgb image showing the current scene from your onboard camera.
                                Numbered lines radiate outward from the target object, with the circle mark indicating the starting point.
                                The black arrows indicate the direction of the direction vectors.
                                The directions are labeled with the numbers: {IDs}.

                                Given the instruction: {INSTRUCTION},
                                step 1: Determine which key part of the object should be manipulated.
                                step 2: 
                                        Directly use the direction markers on the floor 
                                        to identify the direction of this key part relative to the main body of the object.
                                        If the key part is not visible, please try to infer the direction based on the image.
                                        Please do not give me the direction of the target object relative to the current position,
                                        but rather the direction of the key part relative to the main body.
                                step 3: Provide me with a most likely suitable direction number.


                                Note: You can only choose the direction number in the image.
                                At the end, directly return your answer as a JSON in the following format: {{ "points": [] }}
                                Only return the JSON. Do not include explanation or reasoning.
                                """,
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
#                              - Easily finish the task (e.g., pick up the mug) when you are in the candidate base position and facing the target.

#                              - Red arrows on the numbered white circles with a blue outline:
#                                  Use these red arrows to understand the facing direction of the robot when standing at that candidate base position and
#                                  determine whether the candidate base position is suitable for the task and clearly observe the key part of the object required for the task.
# p
#                                 - The red arrows on the numbered white circles with a blue outline: The facing direction of the robot when standing at that candidate base position.
