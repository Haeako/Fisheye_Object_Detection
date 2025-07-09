from ultralytics import YOLO
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval_modified import COCOeval
import json

def f1_score(predictions_path, ground_truths_path):
    coco_gt = COCO(ground_truths_path)

    gt_image_ids = coco_gt.getImgIds()

    with open(predictions_path, 'r') as f:
        detection_data = json.load(f)
    filtered_detection_data = [
        item for item in detection_data if item['image_id'] in gt_image_ids]
    with open('./temp.json', 'w') as f:
        json.dump(filtered_detection_data, f)
    coco_dt = coco_gt.loadRes('./temp.json')
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Assuming the F1 score is at index 20 in the stats array
    return coco_eval.stats[20]  # Return the F1 score from the evaluation stats
    # return 0.85  # Simulated constant value for demo purposes

def get_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    # You need to implement your model here
    model = YOLO(model_path)
    # model.export(format="engine")  # creates 'yolo11n.engine'
    # engine_path = model_path.replace(".pt",".engine")
    # Load the exported TensorRT model
    # trt_model = YOLO(engine_path)

    return model
def preprocess_image(img):
    if img is None:
        raise ValueError("Input image is None.")
    # Preprocess the image for your own model
    return img

import torch

def postprocess_result(results):
    """
    Postprocesses YOLO results to extract boxes, scores, and classes as GPU Tensors.
    
    Args:
        results: The output from a `ultralytics.YOLO` model call.
        
    Returns:
        A list containing [boxes_tensor, scores_tensor, classes_tensor],
        with all tensors remaining on their original GPU device.
    """
    # The 'results' object is a list containing a single 'Results' object for one image
    if not results or not results[0].boxes:
        # Return empty tensors on the same device if no detections
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return [
            torch.empty((0, 4), device=device), 
            torch.empty((0,), device=device), 
            torch.empty((0,), device=device)
        ]
        
    res = results[0] # Get the Results object for the first image
    
    # Extract tensors directly. They are already on the GPU.
    # Do NOT call .cpu() or .tolist() here.
    boxes_tensor = res.boxes.xyxy
    scores_tensor = res.boxes.conf
    classes_tensor = res.boxes.cls
    
    return [boxes_tensor, scores_tensor, classes_tensor]
def changeId(id):
    sceneList = ['M', 'A', 'E', 'N']
    cameraId = int(id.split('_')[0].split('camera')[1])
    sceneId = sceneList.index(id.split('_')[1])
    frameId = int(id.split('_')[2])
    imageId = int(str(cameraId)+str(sceneId)+str(frameId))
    return imageId
