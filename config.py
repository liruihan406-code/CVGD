import torch

args = {
    # Device & Seed
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seed': 42,

    # Dataset & Paths
    'data_path': './data/processed_dataset.h5',
    'teacher_ckpt': './checkpoints/best_teacher.pth',
    'student_ckpt': './checkpoints/best_student.pth',
    'save_dir': './checkpoints',

    # Sequence & Sampling Settings (5Hz standard)
    'in_length': 15,      
    'out_length': 25,     
    'downsample_step': 5, 
    'scale_factor': 10.0,

    # Model Architecture
    'hidden_dim': 128,
    'latent_dim': 16,
    'num_modes': 6,
    'ego_dim': 5,

    # Training Hyperparameters
    'batch_size': 64,
    'lr_teacher': 5e-4,
    'lr_student': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 30,
    'ema_decay': 0.999,
    'grad_clip': 1.0,
    'patience': 15,

    # Distillation Loss Weights
    'w_gt': 1.0,
    'w_feat': 1.0,
    'w_logit': 1.0,
    'w_traj': 1.0,
    'temperature': 2.0,
}
