import torch
import os

RUNTIME = 'TRT'

def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(f"{cameraIndx}{sceneIndx}{frameIndx}")
    return imageId

def convert_box_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def scale_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def preprocess_image(img):
    if img is None:
        raise ValueError("Input image is None.")
    # Preprocess the image for your own model
    return img

def batch_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes in a vectorized way.
        boxes1: [N, 4], boxes2: [M, 4] on the same device.
        Returns: [N, M] IoU matrix.
        """
        # Expand dimensions to broadcast appropriately
        boxes1_expanded = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2_expanded = boxes2.unsqueeze(0)  # [1, M, 4]

        # Intersection
        inter_top_left = torch.max(boxes1_expanded[..., :2], boxes2_expanded[..., :2])
        inter_bottom_right = torch.min(boxes1_expanded[..., 2:], boxes2_expanded[..., 2:])
        inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)
        intersection_area = inter_wh[..., 0] * inter_wh[..., 1]

        # Union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area

        return intersection_area / union_area.clamp(min=1e-6)

def calculate_iou(self, box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    if len(box1) == 4 and len(box2) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    else: return 0.0
    x1_inter = max(x1_1, x1_2); y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2); y2_inter = min(y2_1, y2_2)
    if x2_inter <= x1_inter or y2_inter <= y1_inter: return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1); area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def get_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")