from autodistill_vlpart import VLPart
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
from autodistill_grounding_dino import GroundingDINO

import cv2

# define an ontology to map class names to our VLPart prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundingDINO(ontology=CaptionOntology({"The door pull": "handle"}))

predictions = base_model.predict(
    "/home/zrl/Desktop/research/base_proposal/base_proposal/data/w.png"
)
# print(base_model.class_names)

plot(
    image=cv2.imread(
        "/home/zrl/Desktop/research/base_proposal/base_proposal/data/w.png"
    ),
    classes=["handle"],
    detections=predictions,
)

# label the images in the context_images folder
base_model.label("./context_images", extension=".jpeg")
