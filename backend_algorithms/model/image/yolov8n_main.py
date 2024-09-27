from io import BytesIO
from pathlib import Path

import yaml
import requests
import cv2
import numpy as np
from PIL import Image


def detect(onnx_model, original_image):
    # Load the ONNX model
    model: cv2.dnn.net = cv2.dnn.readNet(onnx_model)

    # Read the input image
    if original_image.shape[-1] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
    else:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    boxes = (np.array(boxes) * scale)[result_boxes].tolist()
    scores = np.array(scores)[result_boxes].tolist()
    class_ids = np.array(class_ids)[result_boxes].tolist()

    return scores, class_ids, boxes


def model_run(
        data
):
    img_url = data["data"]["url"]
    img = np.array(Image.open(BytesIO(requests.get(img_url).content)))

    current_file_path = Path(__file__).resolve()
    classes = yaml.load(
        open(current_file_path.parent.parent / "yamls" / "coco8.yaml", encoding="utf-8"),
        yaml.FullLoader
    )["names"]
    model_path = current_file_path.parent.parent / "weights" / "yolov8n.onnx"

    return {
        "objects": [
            {
                "score": score,
                "label": classes[class_id],
                "contour": {
                    "xmin": box[0],
                    "xmax": box[0] + box[2],
                    "ymin": box[1],
                    "ymax": box[1] + box[3]
                },
                "area": box[2] * box[3]
            }
            for (score, class_id, box) in zip(*detect(str(model_path), img))
        ]
    }
