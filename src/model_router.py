import os
import asyncio
import concurrent.futures
import torch
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import re
import yaml
from utils import RUNTIME

if RUNTIME == 'TRT':
    from trt_agent import TRTInference
elif RUNTIME == 'ONNX':
    from onnx_agent import ONNXInference
else:
    raise ValueError('Runtime not supported')


class ModelEnsembleRouter:
    def __init__(self, models_folder, max_workers=4, config_path='../config/infer_ensemble.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_folder = models_folder
        self.models = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Load config
        self.model_configs = self.config['model_configs']
        self.class_mapping = self.config['class_mapping']
        self.thresholds = self.config['dfine_thresholds_config']
        
    def load_models(self):
        ext = '.engine' if RUNTIME == 'TRT' else '.onnx'
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith(ext)]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_folder, model_file)
            
            # Parse input size from filename
            default_size = self.model_configs.get('dfine', {}).get('imgsz', 640)
            match = re.search(r'(\d+)$', os.path.splitext(model_file)[0])
            input_size = int(match.group(1)) if match and 100 < int(match.group(1)) < 4000 else default_size
            
            # Load model
            if RUNTIME == 'TRT':
                model = TRTInference(model_path, device="cuda:0", max_batch_size=1, 
                                   dfine_input_size=(input_size, input_size))
            else:
                model = ONNXInference(engine_path=model_path, device="cuda:0", max_batch_size=1,
                                    dfine_input_size=(input_size, input_size))
            
            self.models.append(model)
            
        print(f"Loaded {len(self.models)} models")
        
    def predict_single(self, model, img):
        try:
            boxes, scores, classes = model.predict_dfine(img)
            
            if boxes.numel() == 0:
                return [boxes, scores, classes]

            # Apply confidence filtering
            input_size = model.dfine_input_size[0]
            threshold_list = self.thresholds.get(input_size, self.thresholds.get('default'))
            
            if threshold_list:
                thresholds = torch.tensor(threshold_list, device=boxes.device)
                class_indices = torch.clamp(classes.long(), 0, len(thresholds) - 1)
                keep_mask = scores >= thresholds[class_indices]
                boxes, scores, classes = boxes[keep_mask], scores[keep_mask], classes[keep_mask]
            
            return [boxes, scores, classes]
        except Exception as e:
            print(f"Model prediction error: {e}")
            return None

    def weighted_fusion(self, results_list, image_shape, iou_thr=0.5):
        valid_results = [r for r in results_list if r and r[0].numel() > 0]
        if not valid_results:
            return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

        height, width = image_shape[:2]
        boxes_list, scores_list, labels_list = [], [], []

        for boxes, scores, classes in valid_results:
            # Normalize boxes to [0,1]
            boxes_norm = boxes.clone()
            boxes_norm[:, [0, 2]] /= width
            boxes_norm[:, [1, 3]] /= height
            
            boxes_list.append(boxes_norm.cpu().numpy())
            scores_list.append(scores.cpu().numpy())
            labels_list.append(classes.cpu().numpy().astype(int))

        # WBF
        weights = [1] * len(boxes_list)
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights, 
            iou_thr=iou_thr, skip_box_thr=0.1
        )
        
        # Convert back to pixel coords
        fused_boxes = torch.from_numpy(fused_boxes).float()
        fused_boxes[:, [0, 2]] *= width
        fused_boxes[:, [1, 3]] *= height
        
        return (fused_boxes.cuda(), 
                torch.from_numpy(fused_scores).float().cuda(),
                torch.from_numpy(fused_labels).long().cuda())

    async def predict_ensemble(self, processed_img):
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self.executor, self.predict_single, model, processed_img) 
                for model in self.models]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        
        if not valid_results:
            return [[], [], []]
            
        if len(valid_results) == 1:
            boxes, scores, classes = valid_results[0]
        else:
            boxes, scores, classes = self.weighted_fusion(valid_results, processed_img.shape)
        
        return [boxes.cpu().tolist(), scores.cpu().tolist(), classes.cpu().tolist()]
    
    def cleanup(self):
        self.executor.shutdown(wait=True)