import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args
from loader import TrajectoryDataset
from models.teacher_model import TeacherNet
from models.student_model import StudentNet
from train_teacher import compute_rmse


def train():
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    os.makedirs(args['save_dir'], exist_ok=True)
    device = args['device']

    train_set = TrajectoryDataset(args['data_path'], mode='train', step=args['downsample_step'], scale=args['scale_factor'])
    val_set = TrajectoryDataset(args['data_path'], mode='val', step=args['downsample_step'], scale=args['scale_factor'])

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    teacher = TeacherNet(args).to(device)
    teacher.load_state_dict(torch.load(args['teacher_ckpt'], map_location=device))
    teacher.eval()
    teacher.requires_grad_(False)

    student = StudentNet(args).to(device)
    ema_student = copy.deepcopy(student)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args['lr_student'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    time_weights = torch.linspace(1.0, 0.5, steps=args['out_length'], device=device).unsqueeze(0)
    temp = args['temperature']
    best_5s = float('inf')

    for ep in range(args['epochs']):
        student.train()
        pbar = tqdm(train_loader, desc=f"Student Epoch {ep + 1}/{args['epochs']}")

        for hist, fut, ego, neighbors in pbar:
            hist, fut, ego, neighbors = [x.to(device, non_blocking=True) for x in [hist, fut, ego, neighbors]]
            B = hist.shape[0]
            shared_z = torch.randn(B * args['num_modes'], args['latent_dim'], device=device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                t_p, t_feat, t_logit = teacher(hist, ego, neighbors, z=shared_z)
                s_p, s_feat, s_logit = student(hist, ego, neighbors, z=shared_z)

                # 1. Ground-Truth Loss
                dists_gt = torch.norm(s_p - fut.unsqueeze(1), dim=-1).mean(dim=-1)
                l_gt = torch.min(dists_gt, dim=1)[0].mean()

                # 2. Feature-Level Distillation
                s_feat_proj = student.feat_proj(s_feat)
                l_feat = 1.0 - F.cosine_similarity(s_feat_proj, t_feat.detach(), dim=-1).mean()

                # 3. Intention / Logit-Level Distillation
                t_prob = F.softmax(t_logit / temp, dim=-1)
                s_log_prob = F.log_softmax(s_logit / temp, dim=-1)
                l_logit = F.kl_div(s_log_prob, t_prob.detach(), reduction='batchmean') * (temp ** 2)

                # 4. Trajectory-Level Distillation
                with torch.no_grad():
                    t_dists = torch.norm(t_p - fut.unsqueeze(1), dim=-1).mean(dim=-1)
                    best_t_idx = torch.argmin(t_dists, dim=1)
                    best_t_traj = t_p[torch.arange(B, device=device), best_t_idx]

                s_dists = torch.norm(s_p - best_t_traj.unsqueeze(1), dim=-1).mean(dim=-1)
                best_s_idx = torch.argmin(s_dists, dim=1)
                best_s_traj = s_p[torch.arange(B, device=device), best_s_idx]

                raw_traj_loss = F.smooth_l1_loss(best_s_traj, best_t_traj, reduction='none')
                l_traj = (raw_traj_loss.mean(dim=-1) * time_weights).mean()

                loss = (args['w_gt'] * l_gt
                        + args['w_feat'] * l_feat
                        + args['w_logit'] * l_logit
                        + args['w_traj'] * l_traj)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for p, ema_p in zip(student.parameters(), ema_student.parameters()):
                    ema_p.data.mul_(args['ema_decay']).add_(p.data, alpha=1.0 - args['ema_decay'])

            pbar.set_postfix(loss=f"{loss.item():.3f}", gt=f"{l_gt.item():.3f}", feat=f"{l_feat.item():.3f}")

        scheduler.step()

        ema_student.eval()
        val_rmses = []
        with torch.no_grad():
            for hist, fut, ego, neighbors in val_loader:
                hist, fut, ego, neighbors = [x.to(device, non_blocking=True) for x in [hist, fut, ego, neighbors]]
                preds, _, _ = ema_student(hist, ego, neighbors)
                val_rmses.append(compute_rmse(preds, fut, scale=args['scale_factor']))

        avg_rmse = np.mean(val_rmses, axis=0)
        print(f"[Epoch {ep + 1}] RMSE: "
              f"1s={avg_rmse[0]:.3f}, 2s={avg_rmse[1]:.3f}, 3s={avg_rmse[2]:.3f}, 4s={avg_rmse[3]:.3f}, 5s={avg_rmse[4]:.3f}")

        if avg_rmse[4] < best_5s:
            best_5s = avg_rmse[4]
            torch.save(ema_student.state_dict(), args['student_ckpt'])


if __name__ == '__main__':
    train()
