# CVGD: Cognitive Vision-Aware Generative Distillation for Vehicle Trajectory Estimation

## 💡 Motivation & Core Architecture

Human drivers do not merely extrapolate past coordinates; they continuously allocate visual attention to salient traffic participants, reason about kinematic and safety constraints (e.g., TTC, lane offset), and mentally simulate multiple plausible future maneuvers before acting.

**CVGD** bridges cognitive perception and efficient generative modeling through four tightly coupled components:
1. **Cognitive Scene Representation**: Encodes relative kinematics ($\Delta p, \Delta v, \Delta a$) alongside ego safety descriptors ($v, a, r, \text{TTC}, \text{LO}$).
2. **Continuous Active Vision Sector (C-AVS)**: A differentiable, state-conditioned attention sector that dynamically weights surrounding agents without rigid spatial cutoffs.
3. **Cognitive Generative Teacher (LATG)**: Models multi-modal future distributions via Gaussian latent sampling ($z_k \sim \mathcal{N}(0, I)$) and time-conditioned autoregressive displacement synthesis.
4. **Hierarchical Distillation Student**: Transfers decision-level intent (KL divergence), intermediate feature manifolds (Cosine alignment with projection adapters), and temporally weighted trajectory dynamics into a lightweight deployment model.

---

## ⚙️ Installation & Data Preparation

### Environment Setup
```bash
git clone https://github.com/liruihan406-code/CVGD.git
cd CVGD
pip install -r requirements.txt
```

### Data Formatting
The dataloader (`loader.py`) expects preprocessed agent-centric trajectories stored in `.h5` format with the following keys:
- `X`: Historical target coordinates `[N, T_obs, 2]`
- `Y`: Future ground-truth coordinates `[N, T_pred, 2]`
- `ego_states`: Target cognitive state descriptors `[N, 5]` ($v, a, r, \text{TTC}, \text{LO}$)
- `S_delta_p`: Relative neighbor coordinates `[N, num_neighbors, 2]` (padded with `100.0`)

Update `'data_path'` in `config.py` before launching training.

---

## 🚀 Quick Start

### 1. Train the Generative Teacher
Train the teacher model to learn continuous visual attention and multimodal autoregressive trajectory distributions:
```bash
python train_teacher.py
```

### 2. Distill into the Lightweight Student
Freeze the pre-trained teacher (`./checkpoints/best_teacher.pth`) and optimize the student via hierarchical distillation:
```bash
python train_student.py
```

### 3. Benchmark & Evaluation
Measure 1s–5s horizon RMSE (meters) and per-sample inference latency:
```bash
python evaluate_teacher.py
python evaluate_student.py
```

---

## 📝 Citation

If you use this codebase or find our work helpful, please cite:

```bibtex
@article{
}
```
