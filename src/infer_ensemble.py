import os
import time
import json
import argparse
import cv2
import yaml
import asyncio

from utils import preprocess_image, get_image_Id, scale_box, convert_box_to_xywh
from model_router import ModelEnsembleRouter


class Config:
    def __init__(self, config_path="config/infer_ensemble.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


def get_image_files(folder, formats):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if any(f.lower().endswith(ext) for ext in formats)
    ])


def format_predictions(predictions):
    formatted = []
    for image_path, (boxes, scores, classes), (orig_w, orig_h), (new_w, new_h) in predictions:
        image_id = get_image_Id(os.path.basename(image_path))

        scale_x = orig_w / new_w
        scale_y = orig_h / new_h

        for box, score, cls in zip(boxes, scores, classes):
            scaled_box = scale_box(box, scale_x, scale_y)
            formatted.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": convert_box_to_xywh(scaled_box),
                "score": float(score)
            })
    return formatted


async def process_images(router, image_files, config):
    predictions = []
    resize_dims = config.get('inference.resize_dimensions')

    sum_time = 0
    max_fps = 25

    for path in image_files:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}")
            continue
        start = time.time()
        
        orig_h, orig_w = img.shape[:2]

        if resize_dims:
            new_w, new_h = resize_dims
            img = cv2.resize(img, (new_w, new_h))
        else:
            new_w, new_h = orig_w, orig_h


        tensor = preprocess_image(img)
        result = await router.predict_ensemble(path, tensor)

        end = time.time()
        elapsed = end - start
        sum_time += elapsed

        predictions.append((path, result, (orig_w, orig_h), (new_w, new_h)))

    fps = len(image_files) / sum_time if sum_time > 0 else 0
    norm_fps = min(fps, max_fps) / max_fps if max_fps > 0 else 0

    print(f"Total time: {sum_time:.3f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS (max {max_fps}): {norm_fps:.4f}")

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/infer_ensemble.yaml')
    args = parser.parse_args()

    config = Config(args.config)

    image_folder = config.get('data.image_folder')
    formats = config.get('inference.supported_formats', ['.jpg', '.png'])
    image_files = get_image_files(image_folder, formats)

    print(f"Found {len(image_files)} images.")

    models_folder = config.get('models.models_folder')
    router = ModelEnsembleRouter(models_folder, max_workers=config.get('models.max_workers', 4))
    router.load_models()
    print("Ensemble models loaded.")

    predictions = asyncio.run(process_images(router, image_files, config))
    router.cleanup()

    result_json = format_predictions(predictions)
    output_path = config.get('data.output_json', 'output.json')
    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=2)

    print(f"Saved predictions to {output_path}")


if __name__ == '__main__':
    main()
