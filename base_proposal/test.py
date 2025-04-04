from autodistill_grounded_sam import GroundedSAM

from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "cabinet": "cabinet",
            "table": "table",
            "bed": "bed",
            "mug": "mug",
            "bottle": "bottle",
            "book": "book",
            "mouse": "mouse",
            "shelf": "shelf",
            "chair": "chair",
            "desk": "desk",
            "sofa": "sofa",
            "window": "window",
            "television": "television",
        }
    )
)

# run inference on a single image
results = base_model.predict(
    "/home/zrl/Desktop/research/base_proposal/base_proposal/data/w.png"
)

# results = results[results.confidence > 0.5]
# print names of detected classes
class_names = base_model.ontology.classes()
for class_id, conf in zip(results.class_id, results.confidence):
    print(f"{class_names[class_id]}: {conf:.2f}")
plot(
    image=cv2.imread(
        "/home/zrl/Desktop/research/base_proposal/base_proposal/data/w.png"
    ),
    classes=base_model.ontology.classes(),
    detections=results,
)
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
