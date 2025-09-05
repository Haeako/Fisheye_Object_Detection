"""
This code is modified from [https://github.com/Peterande/D-FINE/blob/master/tools/inference/trt_inf.py]
Original author: Peterande
"""

import tensorrt as trt
import torch
import torchvision.transforms.v2 as T
from PIL import Image

import numpy as np
from collections import OrderedDict, namedtuple
    
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
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
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