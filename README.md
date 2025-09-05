# 🚗 Road Object Detection in Fish-Eye Cameras

[🎥 Demo Video](https://github.com/user-attachments/assets/7ca4a06a-4304-4047-b3ab-bd5c720ba268)

*Inference result on **Fisheye1K** using `640x640_fisheye8k.engine` (FP32).*

---

## 👾 System Information

* **Platform**: Jetson AGX Xavier (JetPack 5.1.2, L4T R35.4.1)
* **TensorRT Version**: 8.5.0.2
* **Torch Version**: 2.1.0 (`torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64`)
* **Torchvision Version**: v0.16.1
* **Input Resolution**: 1024×1024 or 640×640

> 🔧 You can adjust parameters via the configuration file: `config/config.yaml`.

---

## 📊 Evaluation Metrics (NVIDIA AI Challenge 2025)

| Model                                                  | AP<sub>0.5:0.95</sub> | AP<sub>0.5</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | F1 Score |
| ------------------------------------------------------ | --------------------- | ---------------- | -------------- | -------------- | -------------- | -------- |
| **1024×1024\_fisheye8k + 1024×1024\_visdra\_m (best)** | 0.5238                | 0.7226           | 0.3369         | 0.6877         | 0.5925         | 0.6139   |
| **640×640\_fisheye8k (FP32)**                          | 0.5556                | 0.7915           | 0.3810         | 0.6880         | 0.5727         | 0.5995   |

---

## ⚡ Inference Speed on Jetson AGX Xavier (30W ALL, single `.engine`)

Results for FP32 are reported in ICCV 2025 evaluation.
FP16 results are shown here for reference only.

| Model                         | FPS   | Normalized (max=25) |
| ----------------------------- | ----- | ------------------- |
| **640×640\_fisheye8k (FP32)** | 12.09 | 0.4836              |
| **640×640\_fisheye8k (FP16)** | 21.89 | 0.8756              |

---

## 📥 Pretrained Weights

⚠️ FP16-trained weights are currently not available.

| Model (FP32)                     | 640×640 Weights                                                                                  | 1024×1024 Weights                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **dfine\_hgnetv2\_m\_fisheye8k** | [Download](https://github.com/Haeako/challenge_iccv_2025/releases/download/V1.0.0/640_fe8k.pth)  | [Download](https://github.com/Haeako/challenge_iccv_2025/releases/download/V1.0.0/last_1024_reduce_carfe8k.pth) |
| **dfine\_hgnetv2\_m\_visdra**    | [Download](https://github.com/Haeako/challenge_iccv_2025/releases/download/V1.0.0/640_indra.pth) | [Download](https://github.com/Haeako/challenge_iccv_2025/releases/download/V1.0.0/last_1024_indra_visdrone.pth) |

---

## 🛠️ TensorRT Engine Build

We use the following command to build `.engine` files:

> ⚠️ Note: Using `--fp16` or `--int8` on FP32-trained models may cause numerical overflow.

```bash
trtexec \
  --onnx=model/dfine_640.onnx \
  --saveEngine=model/dfine_640.engine \
  --memPoolSize=workspace:11000 \
  --useCudaGraph \
  --best \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```

---

## ❤️ Acknowledgement

This work is built upon the amazing [D-FINE](https://github.com/Peterande/D-FINE) project.
---


