import os
import asyncio
import concurrent.futures
import torch
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
import re
import yaml
from utils import RUNTIME

import torchvision
import torchvision.transforms as T

import collections
from collections import OrderedDict
from typing import List, Dict, Any


print(torch.cuda.is_available())

if RUNTIME == 'TRT':
    import tensorrt as trt
    
    class TRTInference(object):
        def __init__(
            self,
            engine_path,
            device="cuda:0",
            backend="torch",
            max_batch_size=1,
            verbose=False,
            dfine_input_size=(640, 640)
        ):
            self.engine_path = engine_path
            self.device = device
            self.backend = backend
            self.max_batch_size = max_batch_size
            self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
            self.engine = self.load_engine(engine_path)
            self.context = self.engine.create_execution_context()
            self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
            self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
            self.input_names = self.get_input_names()
            self.output_names = self.get_output_names()
            
            self.dfine_input_size = dfine_input_size
            self.transform = T.Compose([
                T.Resize(self.dfine_input_size), 
                T.ToTensor()
            ])

        def load_engine(self, path):
            trt.init_libnvinfer_plugins(self.logger, "")
            with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        def get_input_names(self):
            names = []
            for _, name in enumerate(self.engine):
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    names.append(name)
            return names

        def get_output_names(self):
            names = []
            for _, name in enumerate(self.engine):
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    names.append(name)
            return names

        def get_bindings(self, engine, context, max_batch_size=64, device=None) -> OrderedDict:
            Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            bindings = OrderedDict()

            for i, name in enumerate(engine):
                shape = engine.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))

                if shape[0] == -1:
                    shape[0] = max_batch_size
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        context.set_input_shape(name, shape)
                
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

            return bindings

        def run_torch(self, blob):
            for n in self.input_names:
                if blob[n].dtype is not self.bindings[n].data.dtype:
                    blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
                if self.bindings[n].shape != blob[n].shape:
                    self.context.set_input_shape(n, blob[n].shape)
                    self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
                assert self.bindings[n].data.dtype == blob[n].dtype, f"{n} dtype mismatch"
            self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
            self.context.execute_v2(list(self.bindings_addr.values()))
            outputs = {n: self.bindings[n].data for n in self.output_names}
            return outputs

        def __call__(self, blob):
            if self.backend == "torch":
                return self.run_torch(blob)
            else:
                raise NotImplementedError("Only 'torch' backend is implemented.")

        def predict_dfine(self, img):
            """DFINE-specific prediction method that returns GPU tensors."""
            pil_img = Image.fromarray(img)
            orig_size = torch.tensor([[pil_img.size[0], pil_img.size[1]]], device=self.device)
            im_data = self.transform(pil_img).unsqueeze(0).pin_memory().to(self.device, non_blocking=True)

            blob = {"images": im_data, "orig_target_sizes": orig_size}
            torch.cuda.synchronize()
            with torch.no_grad():
                output = self(blob)
                labels = output["labels"][0]
                boxes = output["boxes"][0]
                scores = output["scores"][0]
                return [boxes, scores, labels]

elif RUNTIME == 'ONNX':
    import onnxruntime as ort
    print (ort.get_device())
    class ONNXInference(object):
        def __init__(
            self,
            engine_path,
            device="cuda:0",
            backend="onnx",
            max_batch_size=1,
            verbose=False,
            dfine_input_size=(640, 640)
        ):
            self.engine_path = engine_path
            self.device = device
            self.backend = backend
            self.max_batch_size = max_batch_size
            self.verbose = verbose
            self.dfine_input_size = dfine_input_size
            
            # Set providers based on device
            providers = ['CUDAExecutionProvider'] if 'cuda' in device.lower() else ['CPUExecutionProvider']
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(engine_path, providers=providers)
            
            # Get input and output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            # Create transform
            self.transform = T.Compose([
                T.Resize(self.dfine_input_size), 
                T.ToTensor()
            ])

        @staticmethod
        def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
            """Resizes an image while maintaining aspect ratio and pads it."""
            original_width, original_height = image.size
            ratio = min(size / original_width, size / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            image = image.resize((new_width, new_height), interpolation)

            # Create a new image with the desired size and paste the resized image onto it
            new_image = Image.new("RGB", (size, size))
            new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
            return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2

        def predict_dfine(self, img):
            """DFINE-specific prediction method for ONNX runtime."""
            pil_img = Image.fromarray(img)
            
            # Use aspect ratio preserving resize for ONNX
            resized_img, ratio, pad_w, pad_h = self.resize_with_aspect_ratio(
                pil_img, self.dfine_input_size[0]
            )
            orig_size = torch.tensor([[resized_img.size[1], resized_img.size[0]]])
            
            im_data = self.transform(resized_img).unsqueeze(0)
            
            # Run inference
            output = self.session.run(
                output_names=None,
                input_feed={
                    "images": im_data.numpy(), 
                    "orig_target_sizes": orig_size.numpy()
                },
            )
            
            # Convert outputs to tensors and move to device if CUDA
            labels, boxes, scores = output
            device = self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu'
            
            boxes = torch.from_numpy(boxes).to(device)
            scores = torch.from_numpy(scores).to(device)
            labels = torch.from_numpy(labels).to(device)
            
            return [boxes, scores, labels]

else:
    raise ValueError('Runtime not found, please update "RUNTIME" var. Supported: TRT, ONNX')


class ModelEnsembleRouter:
    def __init__(self, models_folder, max_workers=4, config_path: str = '../config/router.yaml'):
        self.config = self.load_config(config_path)
        self.dfine_models = []
        
        self.models_folder = models_folder
        self.max_workers = self.config.get('max_workers', max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Load configurations from YAML
        self.model_configs = self.config['model_configs']
        self.class_specific_configs = self.config['class_specific_configs']
        self.class_mapping = self.config['class_mapping']
        # Create a reverse mapping for easy name-to-ID lookup
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}

        # Confidence thresholds for DFINE
        self.dfine_thresholds_config = self.config['dfine_thresholds_config']
        
        print(f"🚀 Initialized DFINE-only ensemble router with {RUNTIME} runtime")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
                required_sections = [
                    'model_configs', 'class_specific_configs', 'class_mapping',
                    'dfine_thresholds_config'
                ]
                
                for section in required_sections:
                    if section not in config:
                        raise ValueError(f"Missing section '{section}' in configuration file.")
                        
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")
    
    def load_models(self):
        """Load DFINE models, parsing input size from filenames."""
        if not os.path.exists(self.models_folder):
            raise FileNotFoundError(f"Models folder {self.models_folder} does not exist.")
        
        # Get appropriate file extension based on runtime
        file_extensions = {
            'TRT': '.engine',
            'ONNX': '.onnx'
        }
        extension = file_extensions.get(RUNTIME)
        if not extension:
            raise ValueError(f"Unsupported runtime: {RUNTIME}")
            
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith(extension)]
        
        if not model_files:
            raise ValueError(f"No {extension} model files found in {self.models_folder}")
        
        print(f"Found {len(model_files)} {RUNTIME} model files:")
        loaded_count = 0
        
        for model_file in model_files:
            model_path = os.path.join(self.models_folder, model_file)
            model_name = model_file.lower()
            
            try:
                # Get default size from config
                default_size = self.model_configs.get('dfine', {}).get('imgsz', 640)
                input_size_val = default_size
                
                # Try to parse size from filename (e.g., "dfine_model_800.engine")
                name_without_ext = os.path.splitext(model_name)[0]
                match = re.search(r'(\d+)$', name_without_ext)
                if match:
                    parsed_size = int(match.group(1))
                    # Basic sanity check for the parsed size
                    if 100 < parsed_size < 4000:
                        input_size_val = parsed_size
                
                input_size_tuple = (input_size_val, input_size_val)
                print(f"  📄 Loading DFINE model: {model_file} with input size {input_size_tuple}")
                
                if RUNTIME == 'TRT':
                    model = TRTInference(
                        model_path,
                        device="cuda:0",
                        max_batch_size=1,
                        dfine_input_size=input_size_tuple
                    )
                elif RUNTIME == 'ONNX':
                    model = ONNXInference(
                        engine_path=model_path,
                        device="cuda:0" if torch.cuda.is_available() else "cpu",
                        max_batch_size=1,
                        dfine_input_size=input_size_tuple
                    )
                
                self.dfine_models.append(model)
                loaded_count += 1
                print(f"    ✅ Successfully loaded")
            
            except Exception as e:
                print(f"    ❌ Warning: Could not load model {model_file}: {e}")

        if loaded_count == 0:
            raise ValueError("No models could be loaded successfully")
                
        print(f"\n🎯 Successfully loaded {loaded_count} DFINE models using {RUNTIME} runtime")
        return loaded_count
        
    def predict_single_model(self, model, img, model_type, class_name=None):
        """Wrapper for single DFINE model prediction with integrated filtering."""
        if model is None:
            return None
        
        try:
            boxes, scores, classes = model.predict_dfine(img)
            
            # Return early if no detections
            if not self.valid_check([boxes, scores, classes]):
                return [boxes, scores, classes]

            input_size = model.dfine_input_size[0]
            original_count = boxes.shape[0]

            # Apply size-specific confidence filtering
            threshold_list = self.dfine_thresholds_config.get(
                input_size, 
                self.dfine_thresholds_config.get('default')
            )
            
            if threshold_list:
                device = boxes.device
                thresholds_tensor = torch.tensor(threshold_list, device=device)
                
                class_indices = classes.long()
                # Clamp indices to prevent out-of-bounds errors
                class_indices = torch.clamp(class_indices, 0, len(thresholds_tensor) - 1)

                valid_thresholds = thresholds_tensor[class_indices]
                keep_mask = scores >= valid_thresholds
                
                # Apply the confidence mask
                boxes, scores, classes = boxes[keep_mask], scores[keep_mask], classes[keep_mask]
            
            conf_filtered_count = boxes.shape[0]
            if original_count != conf_filtered_count:
                pass
            # Optional: Apply class-removal filtering if needed
            # This can be configured in the config file
            classes_to_remove = self.model_configs.get('dfine', {}).get('remove_classes', [])
            
            if classes_to_remove:
                for class_name in classes_to_remove:
                    if class_name in self.reverse_class_mapping:
                        class_idx_to_remove = self.reverse_class_mapping[class_name]
                        keep_mask_class = (classes.long() != class_idx_to_remove)
                        boxes, scores, classes = (
                            boxes[keep_mask_class], 
                            scores[keep_mask_class], 
                            classes[keep_mask_class]
                        )

            return [boxes, scores, classes]
                
        except Exception as e:
            print(f"Error in predict_single_model: {e}")
            return None
    
    def perform_nms(self, boxes, scores, classes, iou_threshold):
        """Performs class-agnostic Non-Maximum Suppression."""
        if boxes.numel() == 0:
            return boxes, scores, classes
        
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]

    def perform_wbf(self, results_list, image_shape, iou_thr=0.5, skip_box_thr=0.1):
        """Performs Weighted Boxes Fusion on results from multiple DFINE models."""
        if weighted_boxes_fusion is None:
            raise ImportError("WBF cannot be performed. Please install 'ensemble-boxes'.")

        height, width = image_shape[:2]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Prepare lists for WBF
        boxes_list = []
        scores_list = []
        labels_list = []

        # Add results from each valid DFINE model
        for result in results_list:
            if not self.valid_check(result):
                continue
                
            boxes, scores, classes = result
            
            if boxes.numel() > 0:
                # Normalize boxes to [0, 1] range
                boxes_norm = boxes.clone()
                boxes_norm[:, [0, 2]] /= width
                boxes_norm[:, [1, 3]] /= height
                
                boxes_list.append(boxes_norm.cpu().numpy())
                scores_list.append(scores.cpu().numpy())
                labels_list.append(classes.cpu().numpy().astype(int))

        if not boxes_list:
            # Return empty tensors if no valid results
            return (
                torch.empty((0, 4), device=device), 
                torch.empty(0, device=device), 
                torch.empty(0, device=device)
            )

        # Perform Weighted Boxes Fusion
        # Give equal weight to all models, can be adjusted
        weights = [1] * len(boxes_list) 
        
        fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
        
        # De-normalize fused boxes back to pixel coordinates
        fused_boxes = torch.from_numpy(fused_boxes_norm).float().to(device)
        fused_boxes[:, [0, 2]] *= width
        fused_boxes[:, [1, 3]] *= height
        
        fused_scores = torch.from_numpy(fused_scores).float().to(device)
        fused_labels = torch.from_numpy(fused_labels).long().to(device)

        return fused_boxes, fused_scores, fused_labels

    def combine_multiple_model_results(self, results_list):
        """Combines results from multiple DFINE models."""
        if not results_list:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return [
                torch.empty((0, 4), device=device), 
                torch.empty(0, device=device), 
                torch.empty(0, device=device)
            ]
        
        # Filter out None or invalid results
        valid_results = [res for res in results_list if self.valid_check(res)]

        if not valid_results:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return [
                torch.empty((0, 4), device=device), 
                torch.empty(0, device=device), 
                torch.empty(0, device=device)
            ]

        # Concatenate tensors from all valid results
        all_boxes = torch.cat([res[0] for res in valid_results], dim=0)
        all_scores = torch.cat([res[1] for res in valid_results], dim=0)
        all_classes = torch.cat([res[2] for res in valid_results], dim=0)
        
        return [all_boxes, all_scores, all_classes]

    async def predict_ensemble(self, image_path, processed_img):
        """Main ensemble prediction method for multiple DFINE models."""
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        # Create tasks for all DFINE models
        for i, model in enumerate(self.dfine_models):
            task = loop.run_in_executor(
                self.executor, 
                self.predict_single_model, 
                model, 
                processed_img, 
                f'dfine_{i}'
            )
            tasks.append(task)
        
        # Collect all results
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and invalid results
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"Error in predict_ensemble() on MODEL{i}: {result}")
                continue
            if self.valid_check(result):
                valid_results.append(result)
        
        if not valid_results:
            print("WARN from predict_ensemble(): No valid results from any DFINE model")
            return [[], [], []]       
        # Ensemble strategy based on number of models
        if len(valid_results) == 1:
            # Single model - just apply NMS
            final_boxes, final_scores, final_classes = self.perform_nms(
                *valid_results[0], 
                iou_threshold=0.5
            )
        else:
            # Multiple models - use WBF for ensemble
            final_boxes, final_scores, final_classes = self.perform_wbf(
                valid_results,
                image_shape=processed_img.shape,
                iou_thr=0.5,
                skip_box_thr=0.1
            )
        
        final_detections = final_boxes.shape[0] if final_boxes.numel() > 0 else 0
        print(f"   🎯 Final ensemble result: {final_detections} objects")
        
        # Convert to CPU lists for compatibility
        if final_detections > 0:
            return [
                final_boxes.cpu().tolist(), 
                final_scores.cpu().tolist(), 
                final_classes.cpu().tolist()
            ]
        else:
            return [[], [], []]
    
    def valid_check(self, result):
        """Check if result has valid detections."""
        if result is None or len(result) < 3:
            return False
        
        # For tensors, check numel(). For lists, check len().
        if hasattr(result[2], 'numel'):
            return result[2].numel() > 0
        return result[2] is not None and len(result[2]) > 0
    
    def get_class_names_from_indices(self, class_indices):
        """Convert class indices to class names."""
        if not class_indices:
            return []
            
        class_names = set()
        if isinstance(class_indices, (list, np.ndarray, torch.Tensor)):
            for idx in class_indices:
                class_id = int(idx)
                if class_id in self.class_mapping:
                    class_names.add(self.class_mapping[class_id])
        return list(class_names)
    
    def get_stats(self):
        """Get ensemble statistics."""
        return {
            "runtime": RUNTIME,
            "models_loaded": {
                "dfine_models": len(self.dfine_models),
                "input_sizes": [model.dfine_input_size for model in self.dfine_models]
            },
            "total_models": len(self.dfine_models),
            "ensemble_mode": "DFINE-only",
            "max_workers": self.max_workers
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)