import os
import time
import argparse
import cv2
import json
import asyncio
import yaml
from ultralytics import YOLO

from utils import f1_score, get_model, preprocess_image, postprocess_result, changeId
from model_router import ModelEnsembleRouter

class Config:
    """Configuration handler for YOLO inference"""
    
    def __init__(self, config_path="config/infer_ensemble.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'data.image_folder')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update_from_args(self, args):
        """Update configuration with command line arguments"""
        arg_mappings = {
            'image_folder': 'data.image_folder',
            'models_folder': 'models.models_folder',
            'model_path': 'models.single_model_path',
            'max_fps': 'inference.max_fps',
            'output_json': 'data.output_json',
            'ground_truths_path': 'data.ground_truths_path',
            'use_ensemble': 'inference.use_ensemble',
            'max_workers': 'models.max_workers'
        }
        
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self._set_nested_value(config_path, getattr(args, arg_name))
    
    def _set_nested_value(self, key_path, value):
        """Set nested configuration value"""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


async def process_images_ensemble(ensemble_router, image_files, config):
    """Process all images using ensemble approach"""
    predictions = []
    total_predict_time = 0
    
    print('🚀 Ensemble prediction started')
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        
        # Apply resize if configured
        resize_dims = config.get('inference.resize_dimensions')
        if resize_dims:
            img = cv2.resize(img, tuple(resize_dims))
        
        if img is None:
            print(f"Could not read image {image_path}. Skipping.")
            continue
            
        start_predict_time = time.time()
        
        try:
            processed_img = preprocess_image(img)
            results = await ensemble_router.predict_ensemble(image_path, processed_img)
            predictions.append((image_path, results))
        except Exception as e:
            if config.get('logging.show_detailed_errors', True):
                print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
            predictions.append((image_path, [[], [], []]))
        
        end_time = time.time()
        total_predict_time += (end_time - start_predict_time)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return predictions, total_predict_time, elapsed_time


def process_images_single_model(model, image_files, config):
    """Process images using single model (original approach)"""
    predictions = []
    total_time = 0
    start_time = time.time()
    progress_interval = config.get('inference.progress_interval', 50)
    
    print('📷 Single model prediction started')
    
    for i, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        
        # Apply resize if configured
        resize_dims = config.get('inference.resize_dimensions')
        if resize_dims:
            img = cv2.resize(img, tuple(resize_dims))
        
        if img is None:
            print(f"⚠️  Warning: Could not read image {image_path}. Skipping.")
            continue
            
        t0 = time.time()
        
        try:
            processed_img = preprocess_image(img)
            results = model(processed_img, verbose=False)
            results = postprocess_result(results)
            predictions.append((image_path, results))
        except Exception as e:
            if config.get('logging.show_detailed_errors', True):
                print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
            predictions.append((image_path, [[], [], []]))
        
        t3 = time.time()
        total_time += (t3 - t0)
        
        # Progress indicator
        if (i + 1) % progress_interval == 0:
            avg_time = total_time / (i + 1)
            print(f"📊 Progress: {i + 1}/{len(image_files)} images ({avg_time*1000:.1f}ms avg)")
    
    elapsed_time = time.time() - start_time
    return predictions, total_time, elapsed_time


def convert_box_to_xywh(box):
    """Convert [x1, y1, x2, y2] to [x, y, width, height]"""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def format_predictions_coco_style(predictions, config):
    """Convert predictions to COCO-style list of dicts"""
    formatted = []
    use_filename_as_id = config.get('output.use_filename_as_id', False)
    
    for image_index, (image_path, results) in enumerate(predictions):
        if results is None or len(results) != 3:
            continue

        boxes, scores, classes = results
        if boxes is None or scores is None or classes is None:
            continue

        # Determine image_id
        if use_filename_as_id:
            image_name = os.path.basename(image_path)
            try:
                image_id = int(os.path.splitext(image_name)[0].split('_')[-1])
            except ValueError:
                image_id = image_index
        else:
            image_id = image_index

        for box, score, cls in zip(boxes, scores, classes):
            formatted.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": convert_box_to_xywh(box),
                "score": float(score)
            })

    return formatted


def print_performance_summary(predictions, total_time, elapsed_time, max_fps, mode="single"):
    """Print comprehensive performance summary"""
    print(f"\n{'='*50}")
    print(f"{mode.upper()} MODEL EVALUATION COMPLETE")
    print(f"{'='*50}")
    
    # Basic stats
    total_images = len(predictions)
    successful_predictions = sum(1 for _, results in predictions if results and len(results) >= 3 and results[0])
    failed_predictions = total_images - successful_predictions
    
    print(f"📊 Processing Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Success: {successful_predictions} ({successful_predictions/total_images*100:.1f}%)")
    print(f"   Failed: {failed_predictions}")
    
    # Performance metrics
    fps = total_images / total_time if total_time > 0 else 0
    normfps = min(fps, max_fps) / max_fps if max_fps > 0 else 0
    
    print(f"\n🚀 Performance Metrics:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Normalized FPS: {normfps:.4f}")
    print(f"   Max FPS threshold: {max_fps}")
    return fps, normfps


def get_image_files(image_folder, supported_formats):
    """Get list of image files from folder"""
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if any(f.lower().endswith(fmt) for fmt in supported_formats)
    ])
    return image_files


def validate_paths(config):
    """Validate required paths exist"""
    image_folder = config.get('data.image_folder')
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
    
    if config.get('inference.use_ensemble'):
        models_folder = config.get('models.models_folder')
        if not os.path.exists(models_folder):
            raise FileNotFoundError(f"Models folder {models_folder} does not exist.")
    else:
        model_path = config.get('models.single_model_path')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference with Ensemble Support")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image_folder', type=str, help='Path to image folder')
    parser.add_argument('--models_folder', type=str, help='Path to folder containing all model files')
    parser.add_argument('--model_path', type=str, help='Single model path')
    parser.add_argument('--max_fps', type=float, help='Maximum FPS for evaluation')
    parser.add_argument('--output_json', type=str, help='Output JSON file for predictions')
    parser.add_argument('--ground_truths_path', type=str, help='Path to ground truths JSON file')
    parser.add_argument('--use_ensemble', action='store_true', help='Use ensemble approach')
    parser.add_argument('--max_workers', type=int, help='Maximum number of worker threads')
    
    args = parser.parse_args()

    # Load configuration
    try:
        config = Config(args.config)
        config.update_from_args(args)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Validate paths
    try:
        validate_paths(config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get image files
    image_folder = config.get('data.image_folder')
    supported_formats = config.get('inference.supported_formats', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
    image_files = get_image_files(image_folder, supported_formats)
    
    if not image_files:
        raise ValueError(f"No image files found in {image_folder}")
    print(f"📁 Found {len(image_files)} images in {image_folder}")
    
    # Process images based on mode
    if config.get('inference.use_ensemble'):
        models_folder = config.get('models.models_folder')
        max_workers = config.get('models.max_workers', 4)
        
        print(f"🎯 Using ENSEMBLE mode with models from {models_folder}")
        
        # Initialize ensemble router
        ensemble_router = ModelEnsembleRouter(models_folder, max_workers=max_workers)
        
        try:
            models_loaded = ensemble_router.load_models()
            print(f"✅ Successfully loaded {models_loaded} models")
            
            # Show ensemble stats
            if config.get('logging.show_performance_stats', True):
                stats = ensemble_router.get_stats()
                print(f"📋 Ensemble configuration: {stats}")
            
            # Process images with ensemble
            predictions, total_time, elapsed_time = asyncio.run(
                process_images_ensemble(ensemble_router, image_files, config)
            )
            mode = "ensemble"
        finally:
            ensemble_router.cleanup()
        
    else:
        model_path = config.get('models.single_model_path')
        print(f"📷 Using SINGLE MODEL mode: {model_path}")
        
        # Load single model
        try:
            model = get_model(model_path)
            print(f"✅ Successfully loaded model")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")
        
        # Process images with single model
        predictions, total_time, elapsed_time = process_images_single_model(
            model, image_files, config
        )
        mode = "single"

    # Format and save predictions
    print(f"\n💾 Formatting and saving predictions...")
    predictions_json = format_predictions_coco_style(predictions, config)
    
    output_json = config.get('data.output_json')
    indent = config.get('output.indent_json', 2)
    
    with open(output_json, 'w') as f:
        json.dump(predictions_json, f, indent=indent)
    print(f"✅ Predictions saved to {output_json}")
    
    # Print performance summary
    if config.get('logging.show_performance_stats', True):
        max_fps = config.get('inference.max_fps', 25.0)
        fps, normfps = print_performance_summary(predictions, total_time, elapsed_time, max_fps, mode)
    
    # Calculate F1 score if ground truth is provided
    ground_truths_path = config.get('data.ground_truths_path')
    if ground_truths_path and os.path.exists(ground_truths_path):
        try:
            print(f"\n🎯 Calculating F1 score...")
            f1 = f1_score(output_json, ground_truths_path)
            harmonic_mean = 2 * f1 * normfps / (f1 + normfps) if (f1 + normfps) > 0 else 0
            
            print(f"📊 Accuracy Metrics:")
            print(f"   F1-score: {f1:.4f}")
            print(f"   Harmonic mean (F1 × NormFPS): {harmonic_mean:.4f}")
        except Exception as e:
            print(f"⚠️  Could not calculate F1 score: {e}")
    
    print(f"\n🎉 Processing complete!")


if __name__ == "__main__":
    main()