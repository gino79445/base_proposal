
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

def create_chat_completion(model, messages, temperature=0.5):
    """Creates a chat completion request to OpenAI."""
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)


def determine_base(image_path, part, number_list):
    """Determines which part to grab when picking up the target object."""
    #  and there is a red point mrked on the {part} of the target object.
    #
    #            Step 1: Understand the target and its position
    #            Identify the {part} of the target object in the red frame and note its location in the image relative to the grid and candidate points.



    base64_image = encode_image_to_base64(image_path)
    response = create_chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a robot capable of manipulating objects and moving."},
            {"role": "user", "content": [
                {"type": "text", "text": f"""This image was captured by the camera mounted on you.
                The object in the red frame is your target for operation.
                To determine the most suitable candidate point on the ground you should move to, follow these steps:

                Step 1: Understand the target and its position
                Identify the {part} of the target object in the red frame and note its location in the image relative to the grid and candidate points.

                Step 2: Determine the position opposite the {part} of the target
                Calculate and identify which candidate point directly corresponds to the position of the {part} of the target object.
                the point on the floor should be directly below and ideally vertically aligned with the target.

                Step 3: Evaluate visibility and angular range
                From this candidate point, assess whether facing the target {part} provides the maximum left and right rotational angle,
                allowing you to continuously see the {part} of the target object without any obstruction.
                
                Step 4: Select the best candidate point
                Compare all candidate points and select the single most suitable point that is directly opposite the {part} of the target object.

                Step 5: **Validate against grid and IDs**  
                Verify that the point is labeled with an ID {number_list} in the image.

                Step 6: **Output the result**  
                Once the best candidate point is identified, must only output the result in the following format: {{"id": id}}.
                Please do not include any additional information or context in the output.

                In the image, candidate points are labeled with IDs {number_list}. Please use these references to understand the spatial relationships between the candidate points and objects, helping you determine the optimal choice."""
},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

    )
    #print(response)
    json_response = json.loads(response.choices[0].message["content"])
    json_response = json_response["id"]
    return json_response
#determine_base(path, part_to_grab_str, count_list)
#result = determine_base("./data/result.png", "spine", [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
#print(result)

