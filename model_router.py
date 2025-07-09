import os
import asyncio
import concurrent.futures
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image
from collections import OrderedDict
import collections
import tensorrt as trt
import yaml
from typing import List, Dict, Any
from utils import get_model, postprocess_result
import weighted_boxes_fusion


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(f"{cameraIndx}{sceneIndx}{frameIndx}")
    return imageId

class TRTInference(object):
    # --- MODIFICATION START ---
    def __init__(
        self,
        engine_path,
        device="cuda:0",
        backend="torch",
        max_batch_size=1,
        verbose=False,
        dfine_input_size=(640, 640) # Add new parameter with a default
    ):
    # --- MODIFICATION END ---
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
        
        # --- MODIFICATION START ---
        # Use the provided dfine_input_size to create the transform
        self.dfine_input_size = dfine_input_size
        self.transform = T.Compose([T.Resize(self.dfine_input_size), T.ToTensor()])
        # --- MODIFICATION END ---

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
            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)
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
        device = self.device
        pil_img = Image.fromarray(img)

        orig_size = torch.tensor([[pil_img.size[0], pil_img.size[1]]], device=device)

        im_data = self.transform(pil_img).unsqueeze(0).pin_memory().to(device, non_blocking=True)


        blob = {"images": im_data, "orig_target_sizes": orig_size}
        torch.cuda.synchronize()
        with torch.no_grad():
            output = self(blob)
            labels = output["labels"][0]
            boxes = output["boxes"][0]
            scores = output["scores"][0]
            return [boxes, scores, labels]
        
class ModelEnsembleRouter:
    def __init__(self, models_folder, max_workers=4, config_path: str = 'config/model_router.yaml'):
        # --- MODIFICATION START ---
        self.config = self._load_config(config_path)
        
        self.dfine_models = []
        # --- MODIFICATION END ---
        self.general_model = None
        self.specialized_models = {}  # YOLO specialized models
        self.night_model = None
        
        self.models_folder = models_folder
        self.max_workers = self.config['max_workers'] if 'max_workers' in self.config else max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Model-specific configurations
        # Load configurations from YAML
        self.model_configs = self.config['model_configs']
        self.class_specific_configs = self.config['class_specific_configs']
        self.class_mapping = self.config['class_mapping']
        # Create a reverse mapping for easy name-to-ID lookup.
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}

        # Confidence thresholds for DFINE
        self.dfine_thresholds_config = self.config['dfine_thresholds_config']
        
    # --- HELPER METHOD ---
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
        try:
            with open(config_path , 'r') as file:
                config = yaml.safe_load(file)
                parameter = ['model_configs', 'class_specific_configs', 'class_mapping',
                    'dfine_thresholds_config']
                for section in parameter:
                    if section not in config:
                        raise ValueError(f"Missing section '{section}' in configuration file.")
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")
        
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
    # --- MODIFICATION START ---
    def load_models(self):
        """Load models, parsing input size for DFINE models from their filenames."""
        if not os.path.exists(self.models_folder):
            raise FileNotFoundError(f"Models folder {self.models_folder} does not exist.")
            
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith('')]
        
        if not model_files:
            raise ValueError(f"No .engine model files found in {self.models_folder}")
            
        print(f"Found {len(model_files)} model files:")
        for model_file in model_files:
            print(f"  - {model_file}")
            
        loaded_count = 0
        
        for model_file in model_files:
            model_path = os.path.join(self.models_folder, model_file)
            model_name = model_file.lower()
            
            try:
                # 1. Get default size from config
                default_size = self.model_configs.get('dfine', {}).get('imgsz', 640)
                input_size_val = default_size
                
                # 2. Try to parse size from filename (e.g., "dfine_model_800.engine")
                name_without_ext = os.path.splitext(model_name)[0]
                # Use regex to find the last sequence of digits in the name
                match = re.search(r'(\d+)$', name_without_ext)
                if match:
                    parsed_size = int(match.group(1))
                    # Basic sanity check for the parsed size
                    if 100 < parsed_size < 4000:
                        input_size_val = parsed_size
                
                input_size_tuple = (input_size_val, input_size_val)
                
                print(f"Loading DFINE model: {model_file} with input size {input_size_tuple}")
                
                # 3. Instantiate TRTInference with the determined size
                self.dfine_models.append(
                    TRTInference(
                        model_path,
                        device="cuda:0",
                        max_batch_size=1,
                        dfine_input_size=input_size_tuple
                    )
                )
                loaded_count += 1
                continue
                # --- (Rest of the loading logic remains the same) ---
                if "night" in model_name:
                    print(f"Loading night model: {model_file} (input size: {self.model_configs['night']['imgsz']})")
                    if self.night_model is None:
                        self.night_model = get_model(model_path)
                        loaded_count += 1
                    else:
                        print(f"Warning: Multiple night models found. Only the first one '{os.path.basename(self.night_model.engine_path)}' will be used.")
                    continue

                is_specialized = False
                specialized_classes = ["truck", "bus", "bike", "pedestrian", "motorcycle", "bicycle", "train", "airplane"]
                for class_name in specialized_classes:
                    if class_name in model_name:
                        imgsz = self.class_specific_configs.get(class_name, self.model_configs['specialized'])['imgsz']
                        print(f"Loading specialized YOLO model for {class_name}: {model_file} (input size: {imgsz})")
                        if class_name not in self.specialized_models:
                            self.specialized_models[class_name] = get_model(model_path)
                            loaded_count += 1
                        else:
                             print(f"Warning: Multiple models for class '{class_name}' found. Only the first one will be used.")
                        is_specialized = True
                        break
                if is_specialized:
                    continue
                    
                if "general" in model_name or "yolo" in model_name:
                    print(f"Loading general/YOLO model: {model_file} (input size: {self.model_configs['yolo']['imgsz']})")
                    model = get_model(model_path)
                    self.yolo_models.append(model)
                    if "general" in model_name and self.general_model is None:
                        self.general_model = model
                    loaded_count += 1
                    continue
            
            except Exception as e:
                print(f"Warning: Could not load model {model_file}: {e}")

        if loaded_count == 0:
            raise ValueError("No models could be loaded successfully")
                
        print(f"\nSuccessfully loaded {loaded_count} models:")
        print(f"  - YOLO models: {len(self.yolo_models)} loaded")
        print(f"  - DFINE models: {len(self.dfine_models)} loaded")
        print(f"  - General model (for specialized path): {'✓' if self.general_model else '✗'}")
        print(f"  - Night: {'✓' if self.night_model else '✗'}")
        print(f"  - Specialized: {list(self.specialized_models.keys())}")
        
        return loaded_count
        
    def predict_single_model(self, model, img, model_type, class_name=None):
        """Wrapper for single model prediction with integrated, size-aware filtering."""
        if model is None:
            return None   
        try:
                # --- MODIFICATION START: Integrated DFINE filtering ---
                boxes, scores, classes = model.predict_dfine(img)
                
                # Return early if no detections
                if not self._has_valid_detections([boxes, scores, classes]):
                    return [boxes, scores, classes]

                input_size = model.dfine_input_size[0]
                original_count = boxes.shape[0]

                # 1. Apply size-specific confidence filtering first
                threshold_list = self.dfine_thresholds_config.get(input_size, self.dfine_thresholds_config.get('default'))
                
                if threshold_list:
                    device = boxes.device
                    thresholds_tensor = torch.tensor(threshold_list, device=device)
                    
                    class_indices = classes.long()
                    # Clamp indices to prevent out-of-bounds errors if an unknown class ID appears
                    class_indices = torch.clamp(class_indices, 0, len(thresholds_tensor) - 1)

                    valid_thresholds = thresholds_tensor[class_indices]
                    keep_mask = scores >= valid_thresholds
                    
                    # Apply the confidence mask
                    boxes, scores, classes = boxes[keep_mask], scores[keep_mask], classes[keep_mask]
                
                conf_filtered_count = boxes.shape[0]
                if original_count != conf_filtered_count:
                    print(f"   🔎 Filtered DFINE ({input_size}x{input_size}): Kept {conf_filtered_count} of {original_count} by confidence.")

                # 2. Apply class-removal filtering (e.g., remove 'bike' for 640 models)
                class_to_remove = None
                
                if class_to_remove and class_to_remove in self.reverse_class_mapping:
                    class_idx_to_remove = self.reverse_class_mapping[class_to_remove]
                    
                    keep_mask_class = (classes.long() != class_idx_to_remove)
                    
                    # Apply the class removal mask
                    boxes, scores, classes = boxes[keep_mask_class], scores[keep_mask_class], classes[keep_mask_class]
                    
                    final_count = boxes.shape[0]
                    if conf_filtered_count != final_count:
                        print(f"   🔎 Filtered DFINE ({input_size}x{input_size}): Removed {conf_filtered_count - final_count} '{class_to_remove}' detections.")

                return [boxes, scores, classes]
                # --- MODIFICATION END ---
                
        except Exception as e:
            print(f"Error in model prediction ({model_type}): {e}")
            return None
    
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
    
    def perform_nms(self, boxes, scores, classes, iou_threshold):
        """
        Performs class-agnostic Non-Maximum Suppression on a set of detections.
        
        Args:
            boxes (torch.Tensor): [N, 4] tensor of bounding boxes.
            scores (torch.Tensor): [N] tensor of confidence scores.
            classes (torch.Tensor): [N] tensor of class IDs.
            iou_threshold (float): The IoU threshold for suppression.

        Returns:
            A tuple of (boxes, scores, classes) tensors after NMS.
        """
        if boxes.numel() == 0:
            return boxes, scores, classes
        
        # torchvision.ops.nms is highly optimized and class-agnostic
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]
    def _perform_wbf(self, yolo_result, dfine_result, image_shape, iou_thr=0.5, skip_box_thr=0.1):
        """
        Performs Weighted Boxes Fusion on results from two models.
        
        Args:
            yolo_result: A tuple of (boxes, scores, classes) from YOLO.
            dfine_result: A tuple of (boxes, scores, classes) from DFINE.
            image_shape: The (height, width) of the original image for normalization.
            iou_thr: IoU threshold for WBF.
            skip_box_thr: Confidence threshold to discard boxes before fusion.
        
        Returns:
            A tuple of (boxes, scores, classes) tensors after WBF.
        """
        if weighted_boxes_fusion is None:
            raise ImportError("WBF cannot be performed. Please install 'ensemble-boxes'.")

        yolo_boxes, yolo_scores, yolo_classes = yolo_result
        dfine_boxes, dfine_scores, dfine_classes = dfine_result
        
        device = yolo_boxes.device
        height, width = image_shape[:2]

        # Prepare lists for WBF
        boxes_list = []
        scores_list = []
        labels_list = []

        # Normalize and add YOLO results
        if yolo_boxes.numel() > 0:
            # Normalize boxes to [0, 1] range
            yolo_boxes_norm = yolo_boxes.clone()
            yolo_boxes_norm[:, [0, 2]] /= width
            yolo_boxes_norm[:, [1, 3]] /= height
            boxes_list.append(yolo_boxes_norm.cpu().numpy())
            scores_list.append(yolo_scores.cpu().numpy())
            labels_list.append(yolo_classes.cpu().numpy())
        
        # Normalize and add DFINE results
        if dfine_boxes.numel() > 0:
            # Normalize boxes to [0, 1] range
            dfine_boxes_norm = dfine_boxes.clone()
            dfine_boxes_norm[:, [0, 2]] /= width
            dfine_boxes_norm[:, [1, 3]] /= height
            boxes_list.append(dfine_boxes_norm.cpu().numpy())
            scores_list.append(dfine_scores.cpu().numpy())
            labels_list.append(dfine_classes.cpu().numpy().astype(int))

        if not boxes_list:
            return torch.empty((0, 4), device=device), torch.empty(0, device=device), torch.empty(0, device=device)

        # Perform Weighted Boxes Fusion
        # Weights can be adjusted, e.g., [2, 1] to give more weight to YOLO
        # For now, we give them equal weight.
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
    # --- NEW HELPER METHOD 2 ---
    def _filter_dfine_by_confidence(self, boxes, scores, classes):
        """Applies class-specific confidence thresholds to DFINE results."""
        if boxes.numel() == 0:
            return boxes, scores, classes

        device = boxes.device
        dfine_thresholds_tensor = torch.tensor(self.dfine_thresholds, device=device)
        
        # Get the threshold for each detection based on its class
        class_indices = classes.long()
        valid_thresholds = dfine_thresholds_tensor[class_indices]
        
        # Create a mask to keep only detections above their class-specific threshold
        keep_mask = scores >= valid_thresholds
        
        return boxes[keep_mask], scores[keep_mask], classes[keep_mask]


    # --- REWRITTEN ENSEMBLE METHOD ---
    # In ModelEnsembleRouter class, modify this method

    # --- MODIFIED ENSEMBLE METHOD ---
    def ensemble(self, yolo_result, dfine_result, image_shape):
        """
        Ensembles YOLO and DFINE results using a 2-stage strategy with WBF.
        Filtering is now done within predict_single_model.

        Stage 1: Clean up YOLO detections with NMS.
        Stage 2: Clean up DFINE detections with NMS.
        Stage 3: Merge both sets using Weighted Boxes Fusion (WBF).
        """
        yolo_boxes, yolo_scores, yolo_classes = yolo_result
        dfine_boxes, dfine_scores, dfine_classes = dfine_result

        # --- Stage 1: Intra-Model NMS for YOLO results ---
        print(f"   -> Stage 1: Cleaning YOLO results (IoU > 0.8)")
        cleaned_yolo_boxes, cleaned_yolo_scores, cleaned_yolo_classes = self.perform_nms(
            yolo_boxes, yolo_scores, yolo_classes, iou_threshold=0.8
        )
        print(f"      YOLO objects after cleanup: {cleaned_yolo_boxes.shape[0]}")

        # --- Stage 2: Intra-Model NMS for DFINE results ---
        # --- MODIFICATION: The confidence filter is no longer called here ---
        print(f"   -> Stage 2: Cleaning DFINE results (IoU > 0.8)")
        cleaned_dfine_boxes, cleaned_dfine_scores, cleaned_dfine_classes = self.perform_nms(
            dfine_boxes, dfine_scores, dfine_classes, iou_threshold=0.8
        )
        print(f"      DFINE objects after cleanup: {cleaned_dfine_boxes.shape[0]}")
        
        # --- Stage 3: Final Inter-Model Ensemble Merge using WBF ---
        print(f"   -> Stage 3: Performing final ensemble merge using WBF (IoU > 0.5)")
        
        final_boxes, final_scores, final_classes = self._perform_wbf(
            (cleaned_yolo_boxes, cleaned_yolo_scores, cleaned_yolo_classes),
            (cleaned_dfine_boxes, cleaned_dfine_scores, cleaned_dfine_classes.long()),
            image_shape=image_shape,
            iou_thr=0.5,
            skip_box_thr=0.1
        )
        
        print(f"      Total objects after WBF: {final_boxes.shape[0]}")

        return [final_boxes.cpu().tolist(), final_scores.cpu().tolist(), final_classes.cpu().tolist()]

    # --- NEW HELPER FUNCTION ---
    def _combine_multiple_model_results(self, results_list):
        """
        Combines results from multiple models.
        Args:
            results_list: A list of prediction results, e.g., [[boxes1, scores1, classes1], [boxes2, scores2, classes2]]
        Returns:
            A single combined result: [all_boxes, all_scores, all_classes] as GPU tensors.
        """
        if not results_list:
            # Return empty tensors on the GPU if no results
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return [torch.empty((0, 4), device=device), torch.empty(0, device=device), torch.empty(0, device=device)]
        
        # Filter out None or invalid results
        valid_results = [res for res in results_list if self._has_valid_detections(res)]

        if not valid_results:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return [torch.empty((0, 4), device=device), torch.empty(0, device=device), torch.empty(0, device=device)]

        # Concatenate tensors from all valid results
        all_boxes = torch.cat([res[0] for res in valid_results], dim=0)
        all_scores = torch.cat([res[1] for res in valid_results], dim=0)
        all_classes = torch.cat([res[2] for res in valid_results], dim=0)
        
        return [all_boxes, all_scores, all_classes]

    # --- MODIFICATION START ---
    async def predict_ensemble(self, image_path, processed_img):
        """Main ensemble prediction method, adapted for multiple YOLO/DFINE models."""
        
        # Determine if night model should be used based on filename
        image_filename = os.path.basename(image_path).upper()
        use_night_model = "N_" in image_filename
        
        loop = asyncio.get_event_loop()
        
        # Check if we should use the YOLO+DFINE ensemble path
        # This path is used if we have at least one YOLO and one DFINE model, and no specialized models.
        use_yolo_dfine_path = True

        if use_yolo_dfine_path:
            print(f"🤖 Running DFINE ({len(self.dfine_models)} models) ensemble mode")
            
            tasks = []
            
            # Create tasks for all DFINE models
            for model in self.dfine_models:
                task = loop.run_in_executor(
                    self.executor, self.predict_single_model, model, processed_img, 'dfine'
                )
                tasks.append(('dfine', task))
            
            # Collect all results
            all_raw_results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
            
            # Separate results by type
            dfine_results = []
            for i, (model_type, _) in enumerate(tasks):
                result = all_raw_results[i]
                if isinstance(result, Exception):
                    print(f"Error in {model_type} model: {result}")
                    continue
                elif model_type == 'dfine':
                    dfine_results.append(result)
            
            # Combine results from multiple models of the same type
            combined_dfine_result = self._combine_multiple_model_results(dfine_results)
            
            dfine_detections = combined_dfine_result[0].shape[0] if combined_dfine_result else 0
            
            print(f"   📊 Combined DFINE detected: {dfine_detections} objects")
            
            # Ensemble the combined DFINE results
            final_result = self.ensemble( 
                combined_dfine_result,
                image_shape=processed_img.shape # Pass H, W, C
            )
            final_detections = len(final_result[0]) if self._has_valid_detections(final_result) else 0
            print(f"   🎯 Final ensemble result: {final_detections} objects")
            
            return final_result
        
        else:
            # Fallback to the original logic for specialized models
            # This part remains mostly unchanged, using self.general_model
            print("Legacy ensemble mode (specialized/night models)")
            results = {"general": None, "specialized": {}, "night": None}
            # Stage 1: Always run general model first (if available)
            if self.general_model:
                results["general"] = await loop.run_in_executor(
                    self.executor, self.predict_single_model, self.general_model, processed_img, 'general'
                )
            
            # Stage 2: Determine which specialized models to run
            specialized_tasks = []
            detected_classes = []
            if results["general"] and self._has_valid_detections(results["general"]):
                classes = results["general"][2]
                detected_classes = self.get_class_names_from_indices(classes)
                for class_name in detected_classes:
                    if class_name in self.specialized_models:
                        task = loop.run_in_executor(
                            self.executor, self.predict_single_model, self.specialized_models[class_name], 
                            processed_img, 'specialized', class_name
                        )
                        specialized_tasks.append((class_name, task))
            
            # Stage 3: Run night model if needed
            night_task = None
            if use_night_model and self.night_model:
                night_task = loop.run_in_executor(
                    self.executor, self.predict_single_model, self.night_model, processed_img, 'night'
                )
            
            # Collect all results
            for class_name, task in specialized_tasks:
                try: results["specialized"][class_name] = await task
                except Exception as e: print(f"Error in specialized model {class_name}: {e}")
            
            if night_task:
                try: results["night"] = await night_task
                except Exception as e: print(f"Error in night model: {e}")

            return self.combine_results(results, use_night_model, detected_classes)
    # --- MODIFICATION END ---
    
    def _has_valid_detections(self, result):
        """Check if result has valid detections"""
        if result is None or len(result) < 3: return False
        # For tensors, check numel(). For lists, check len().
        if hasattr(result[2], 'numel'):
            return result[2].numel() > 0
        return result[2] is not None and len(result[2]) > 0
    
    def get_class_names_from_indices(self, class_indices):
        """Convert class indices to class names"""
        if not class_indices: return []
        class_names = set()
        if isinstance(class_indices, (list, np.ndarray, torch.Tensor)):
            for idx in class_indices:
                class_id = int(idx)
                if class_id in self.class_mapping:
                    class_names.add(self.class_mapping[class_id])
        return list(class_names)
    
    def combine_results(self, results, use_night_priority, detected_classes):
        """Combine results from different models with priority logic"""
        detection_counts = {}
        if results["specialized"] and any(self._has_valid_detections(r) for r in results["specialized"].values()):
            final_result = self.merge_specialized_with_general(results["general"], results["specialized"], detected_classes)
            detection_counts["specialized"] = len(final_result[0]) if final_result[0] is not None else 0
            active_specialized = [k for k, v in results["specialized"].items() if self._has_valid_detections(v)]
            print(f"🎯 Using specialized models {active_specialized} + general (detected: {detection_counts['specialized']} objects)")
        elif self._has_valid_detections(results["general"]):
            final_result = results["general"]
            detection_counts["general"] = len(final_result[0]) if final_result[0] is not None else 0
            print(f"🔍 Using general model only (detected: {detection_counts['general']} objects)")
        else:
            final_result = [[], [], []]
            print("⚠️  No valid results from any model")
        return final_result
    
    def merge_specialized_with_general(self, general_result, specialized_results, detected_classes):
        """Merge specialized model results with general model results"""
        if not self._has_valid_detections(general_result):
            for spec_result in specialized_results.values():
                if self._has_valid_detections(spec_result): return spec_result
            return [[], [], []]
            
        merged_boxes = list(general_result[0]); merged_scores = list(general_result[1]); merged_classes = list(general_result[2])
        enhancements_made = []
        for class_name, specialized_result in specialized_results.items():
            if self._has_valid_detections(specialized_result) and class_name in detected_classes:
                if specialized_result[0]:
                    merged_boxes.extend(specialized_result[0])
                    merged_scores.extend(specialized_result[1])
                    merged_classes.extend(specialized_result[2])
                    enhancements_made.append(class_name)
        if enhancements_made:
            print(f"Enhanced with specialized models: {enhancements_made}")
        return [merged_boxes, merged_scores, merged_classes]
    
    def get_stats(self):
        """Get ensemble statistics"""
        return {
            "models_loaded": {
                "dfine_models": len(self.dfine_models),
                "general_model_assigned": self.general_model is not None,
                "night_model_assigned": self.night_model is not None,
                "specialized": list(self.specialized_models.keys())
            },
            "total_models": (
                len(self.yolo_models) + len(self.dfine_models) +
                (1 if self.night_model else 0) + len(self.specialized_models)
            ),
            "ensemble_mode": "YOLO+DFINE" if (self.yolo_models and self.dfine_models and not self.specialized_models) else "Specialized/Legacy"
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)