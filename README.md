# CVGD: Cognitive Vision-Aware Generative Distillation for Vehicle Trajectory Estimation

## 💡 Motivation & Core Architecture

Human drivers do not merely extrapolate past coordinates; they continuously allocate visual attention to salient traffic participants, reason about kinematic and safety constraints (e.g., TTC, lane offset), and mentally simulate multiple plausible future maneuvers before acting.

**CVGD** bridges cognitive perception and efficient generative modeling through four tightly coupled components:
1. **Cognitive Scene Representation**: Encodes relative kinematics ($\Delta p, \Delta v, \Delta a$) alongside ego safety descriptors ($v, a, r, \text{TTC}, \text{LO}$).
2. **Continuous Active Vision Sector (C-AVS)**: A differentiable, state-conditioned attention sector that dynamically weights surrounding agents without rigid spatial cutoffs.
3. **Cognitive Generative Teacher (LATG)**: Models multi-modal future distributions via Gaussian latent sampling ($z_k \sim \mathcal{N}(0, I)$) and time-conditioned autoregressive displacement synthesis.
4. **Hierarchical Distillation Student**: Transfers decision-level intent (KL divergence), intermediate feature manifolds (Cosine alignment with projection adapters), and temporally weighted trajectory dynamics into a lightweight deployment model.

<p align="center">
  <img src="pic/Figure1.png" width="85%" alt="CVGD Framework Overview">
</p>

---

## 📂 Repository Structure

```text
CVGD/
├── config.py               # Centralized hyperparameters, sequence horizons (5Hz), and loss weights
├── loader.py               # HDF5 trajectory dataset loader with 25Hz -> 5Hz temporal alignment
├── models/
│   ├── __init__.py
│   ├── teacher_model.py    # C-AVS module & Latent Autoregressive Trajectory Generator (LATG)
│   └── student_model.py    # Compact student network & non-linear feature projection adapter
├── train_teacher.py        # Stage 1: Teacher training with time-weighted Smooth-L1 & diversity annealing
├── evaluate_teacher.py     # Teacher evaluation (1s-5s Best-of-K RMSE & latency profiling)
├── train_student.py        # Stage 2: Joint multi-level knowledge distillation (GT + Feat + Logit + Traj)
├── evaluate_student.py     # Student evaluation & deployment efficiency benchmark
├── requirements.txt        # Python package dependencies
└── pic/                    # Figures and qualitative trajectory visualizations
```

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
@article{li2026cvgd,
  title={Cognitive Vision-Aware Generative Distillation Framework for Robust Vehicle Trajectory Estimation},
  author={Li, Ruihan and et al.},
  journal={Under Review},
  year={2026}
}
```
