"""
This code is modified from [https://github.com/Peterande/D-FINE/blob/master/tools/inference/onnx_inf.py]
Original author: Peterande
"""

import onnxruntime as ort
import torch
from PIL import Image
import torchvision.transforms.v2 as T
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