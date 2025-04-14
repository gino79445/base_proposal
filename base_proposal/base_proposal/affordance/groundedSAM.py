# import os
# import cv2
# import torch
# import numpy as np
# from autodistill.detection import CaptionOntology
# from autodistill_grounding_dino import GroundingDINO as DINO
# from segment_anything import sam_model_registry, SamPredictor
# from segment_anything.utils.transforms import ResizeLongestSide
# import gc
#
#
# def detect_and_segment(
#    image_path,
#    object_name,
#    sam_checkpoint="./base_proposal/affordance/sam_vit_h.pth",
#    output_path="./data/segmented.png",
# ):
#    os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#    # 初始化 GroundingDINO
#    ontology = CaptionOntology({object_name: object_name})
#    dino_model = DINO(ontology=ontology)
#    detections = dino_model.predict(image_path)
#
#    # 載入圖像與 SAM 模型
#    image = cv2.imread(image_path)
#    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
#    sam.to("cuda")
#    predictor = SamPredictor(sam)
#    predictor.set_image(image_rgb)
#
#    results = []
#
#    for class_id, conf, box in zip(
#        detections.class_id, detections.confidence, detections.xyxy
#    ):
#        x1, y1, x2, y2 = map(int, box)
#        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
#        transform = ResizeLongestSide(sam.image_encoder.img_size)
#        transformed_box = transform.apply_boxes(input_box, image.shape[:2])
#
#        # 再把這個丟進去 SAM
#        input_box_torch = torch.tensor(transformed_box, device="cuda")
#
#        # 多遮罩輸出 + 合併遮罩
#        masks, scores, _ = predictor.predict_torch(
#            point_coords=None,
#            point_labels=None,
#            boxes=input_box_torch,
#            multimask_output=True,
#        )
#        merged_mask = torch.any(masks[0], dim=0).cpu().numpy()
#
#        # 半透明遮罩合成
#        alpha = 0.5
#        overlay = np.zeros_like(image)
#        overlay[merged_mask] = (0, 125, 0)  # 綠色遮罩
#        overlay = (overlay * alpha + image * (1 - alpha)).astype(np.uint8)
#        image[merged_mask] = overlay[merged_mask]
#
#        # 繪製 bbox 與標籤
#        label = ontology.classes()[class_id]
#        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
#        cv2.putText(
#            image,
#            label,
#            (x1, max(y1 - 10, 15)),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.6,
#            (255, 0, 0),
#            1,
#        )
#
#        results.append(
#            {
#                "label": label,
#                "confidence": float(conf),
#                "bbox": [x1, y1, x2, y2],
#                "mask": merged_mask,
#            }
#        )
#
#    # 儲存可視化圖像
#    cv2.imwrite(output_path, image)
#    print(f"✅ Segmentation result saved to {output_path}")
#
#    # save masks with most confidence
#    for i, result in enumerate(results):
#        mask = result["mask"]
#        mask_output_path = os.path.join(os.path.dirname(output_path), "mask.png")
#        cv2.imwrite(mask_output_path, mask.astype(np.uint8) * 255)
#        print(f"✅ Mask saved to {mask_output_path}")
#        break
#
#    del dino_model
#    del predictor
#    del sam
#
#    gc.collect()
#    torch.cuda.empty_cache()
#    return results
#

# if __name__ == "__main__":
#    result = detect_and_segment("./data/rgb.png", "black mug")
#    for r in result:
#        print(f"{r['label']} @ {r['bbox']} conf={r['confidence']:.2f}")
#
#        print(f"mask shape: {r['mask'].shape}")


import os
import cv2
import torch
import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO as DINO
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import gc

# ✅ 全局模型快取
_dino_model = None
_sam_predictor = None
_transform = None


def load_models(object_name, sam_checkpoint):
    global _dino_model, _sam_predictor, _transform

    if _dino_model is None:
        ontology = CaptionOntology({object_name: object_name})
        _dino_model = DINO(ontology=ontology)

    if _sam_predictor is None:
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to("cuda")
        _sam_predictor = SamPredictor(sam)
        _transform = ResizeLongestSide(sam.image_encoder.img_size)


def detect_and_segment(
    image_path,
    object_name,
    sam_checkpoint="./base_proposal/affordance/sam_vit_h.pth",
    output_path="./data/segmented.png",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ 載入模型（只會做一次）
    load_models(object_name, sam_checkpoint)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _sam_predictor.set_image(image_rgb)
    detections = _dino_model.predict(image_path)

    results = []

    for class_id, conf, box in zip(
        detections.class_id, detections.confidence, detections.xyxy
    ):
        x1, y1, x2, y2 = map(int, box)
        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        transformed_box = _transform.apply_boxes(input_box, image.shape[:2])
        input_box_torch = torch.tensor(transformed_box, device="cuda")

        masks, scores, _ = _sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_box_torch,
            multimask_output=True,
        )
        merged_mask = torch.any(masks[0], dim=0).cpu().numpy()

        alpha = 0.5
        overlay = np.zeros_like(image)
        overlay[merged_mask] = (0, 125, 0)
        overlay = (overlay * alpha + image * (1 - alpha)).astype(np.uint8)
        image[merged_mask] = overlay[merged_mask]

        label = _dino_model.ontology.classes()[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1,
        )

        results.append(
            {
                "label": label,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "mask": merged_mask,
            }
        )

    cv2.imwrite(output_path, image)
    print(f"✅ Segmentation result saved to {output_path}")

    for i, result in enumerate(results):
        mask = result["mask"]
        mask_output_path = os.path.join(os.path.dirname(output_path), "mask.png")
        cv2.imwrite(mask_output_path, mask.astype(np.uint8) * 255)
        print(f"✅ Mask saved to {mask_output_path}")
        break

    return results
