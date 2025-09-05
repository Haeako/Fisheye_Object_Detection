# 🚗 Road Object Detection in Fish-Eye Cameras (ICCV Challenge 2025)

[fish-eye-demo.webm](https://github.com/user-attachments/assets/7ca4a06a-4304-4047-b3ab-bd5c720ba268)

<sub>Inference result on Fisheye1K using 640x640_fisheye8k+visdra.engine (FP32)</sub>

---

## 📊 System Information
- **Platform**: Jetson AGX Xavier (JetPack 5.1.2 (L4T R35.4.1))  
- **TensorRT Version**: 8.5.0.2  
- **Torch Version**: 2.1.0 (torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64)  
- **Torchvision Version**: –  
- **Input Resolution**: 1024×1024 or 640×640  

!Change configuration file (`config/config.yaml`) to adjust parameters.

---

## 📊 Evaluation Metrics (from NVIDIA AI Challenge 2025)

| Model | AP<sub>0.5:0.95</sub> | AP<sub>0.5</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | F1 Score |
|-------------------------------|------------------|----------------|----------------|----------------|----------|
| [fuse ]1024x1024_fisheye8k + 1024x1024_visdra_m (best) | 0.5238 | 0.7226 | 0.3369 | 0.6877 | 0.5925 | 0.6139 |
| 640x640_fisheye8k+visdra (fp32) | 0.5556 | 0.7915 | 0.3810 | 0.6880 | 0.5727 | 0.5995 |

---

### Inference Speed on Jetson AGX Xavier (30W ALL, single `.engine`)

| DEVICE | FPS | Normalize (max25) |
|-------------------------|------|-------------------|
| 640x640_fisheye8k+visdra (FP32) | 12.09 | 0.4836 |


---

## Pretrained weights  

| Model (fp32) | Input Size | 640x640 Weights|1024x1024 Weights |
|-------|------------|----------------|------------------|
| dfine_hgnetv2_m_fisheye8k  | [Download](link_2_pth) |[Download] |
| dfine_hgnetv2_m_visdra     | [Download](link_1_pth) |[Download] |
|dfine_hgnetv2_m_fisheye8k+visdra| [Download](link_2_pth)|  |

---

## TensorRT Engine build command

```bash
trtexec \
  --onnx=model/dfine_640.onnx \
  --saveEngine=model/dfine_640.engine \
  --memPoolSize=workspace:11000 \
  --useCudaGraph \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```
