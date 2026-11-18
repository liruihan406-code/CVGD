🚗 CVGD-HLTP: Cognitive Vision-Aware Generative Distillation for Human-Like Trajectory Prediction

This repository contains the official implementation of **Cognitive Vision-Aware Generative Distillation Framework for Robust Vehicle Trajectory Estimation**, tailored for advanced autonomous driving and intelligent transportation systems.

---

## ⚠️ **Project Status**

🚀 **Core Architecture Uploaded!**  
We have released the complete implementation, including the **Continuous Active Vision Sector (C-AVS)**, **Latent Autoregressive Trajectory Generator (LATG)**, and **Hierarchical Knowledge Distillation** framework. You can seamlessly replicate the multi-modal trajectory prediction results across NGSIM, nuScenes, and RoundD benchmarks.

---

## 📖 Overview

Accurate vehicle trajectory prediction is essential for autonomous driving and safety-critical decision-making. Traditional models often treat perception and trajectory generation as decoupled modules or rely heavily on static map priors. 

Inspired by human cognitive processes, our framework—**Cognitive Vision-Aware Generative Distillation (CVGD)**—simulates how a driver continuously observes visually salient traffic regions, reasons about scene constraints, and forms multiple plausible future intentions. 

---

## ✨ Highlights

- **Continuous Active Vision Sector (C-AVS)**  
  Dynamically adapts attention allocation over surrounding traffic participants, mimicking human selective visual attention under complex driving dynamics.

- **Latent Autoregressive Trajectory Generator (LATG)**  
  Combines continuous latent variable sampling with autoregressive trajectory synthesis to model multimodal future distributions with high temporal consistency.

- **Hierarchical Knowledge Distillation**  
  Transfers knowledge from the high-capacity teacher to a lightweight student model across three levels: **Decision (Logit)**, **Feature Flow**, and **Trajectory Geometry**, preserving predictive capability while drastically reducing inference complexity.

- **Superior Robustness**  
  Extensively evaluated on high-density highway (NGSIM), complex urban (nuScenes), and roundabout (RoundD) benchmarks, demonstrating outstanding precision and deployment efficiency.

---

## 🧠 Framework Architecture

- **Teacher Model**: Combines motion encoding, C-AVS interaction aggregation, and the Latent Autoregressive Trajectory Generator (LATG) to yield diverse, multimodal future hypotheses.
- **Student Model**: Implements a compact architecture optimized via multi-stage hierarchical supervision to inherit the teacher's cognitive and distributional reasoning.

---

## ⚙️ Environment Requirements

- **Operating System**: Ubuntu 20.04 / Google Colab (A100 GPU recommended)
- **CUDA Version**: 11.3+
- **Python**: 3.8+

---

## 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/liruihan406-code/CVGD.git](https://github.com/liruihan406-code/CVGD.git)
   cd CVGD
