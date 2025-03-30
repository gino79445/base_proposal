import requests
from PIL import Image, ImageDraw, ImageFont
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/home/zrl/Pictures/Screenshots/d.png").convert("RGB")


# Define text labels
text_labels = [["The door pull of the cabinet"]]
inputs = processor(text=text_labels, images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert outputs to bounding boxes
target_sizes = torch.tensor([(image.height, image.width)])
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
)

# Retrieve results
result = results[0]
boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]

# Draw bounding boxes
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except IOError:
    font = ImageFont.load_default()

for box, score, text_label in zip(boxes, scores, text_labels):
    box = [round(i, 2) for i in box.tolist()]
    label = f"{text_label} ({round(score.item(), 2)})"
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1] - 10), label, fill="red", font=font)
    print(
        f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}"
    )

# Show image with detections
image.show()
