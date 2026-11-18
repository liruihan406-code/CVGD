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


def teacher_loss(preds, gt, current_epoch, total_epochs):
    # preds: [B, K, T, 2], gt: [B, T, 2]
    B, K, T, _ = preds.shape
    gt_expanded = gt.unsqueeze(1)
    t_weights = torch.linspace(1.0, 3.0, T, device=preds.device).view(1, 1, T)

    dist = torch.norm(preds - gt_expanded, dim=-1)
    sum_weighted_dist = torch.sum(dist * t_weights, dim=-1)
    best_mode = torch.argmin(sum_weighted_dist, dim=1)
    best_preds = preds[torch.arange(B, device=preds.device), best_mode]

    l1_loss = F.smooth_l1_loss(best_preds, gt, reduction='none', beta=0.5)
    l_var = (torch.sum(l1_loss, dim=-1) * t_weights.squeeze(1)).mean()

    progress = current_epoch / total_epochs
    if progress < 0.2:
        div_weight = 0.0
    elif progress < 0.7:
        div_weight = ((progress - 0.2) / 0.5) * 0.05
    else:
        div_weight = 0.05

    endpoints = preds[:, :, -1, :]
    diff = endpoints.unsqueeze(2) - endpoints.unsqueeze(1)
    dist_endpoints = torch.norm(diff, dim=-1) + torch.eye(K, device=preds.device).unsqueeze(0) * 1e4
    min_dist, _ = torch.min(dist_endpoints, dim=-1)
    l_div = torch.clamp(0.05 - min_dist, min=0.0).mean()

    return l_var + div_weight * l_div


def compute_rmse(preds, gt, scale=10.0):
    fde = torch.norm(preds[:, :, -1, :] * scale - gt[:, -1, :].unsqueeze(1) * scale, dim=-1)
    best_idx = torch.argmin(fde, dim=1)
    best_traj = preds[torch.arange(preds.shape[0], device=preds.device), best_idx] * scale
    gt_scaled = gt * scale

    indices = [4, 9, 14, 19, 24]
    return [torch.sqrt(torch.mean(torch.norm(best_traj[:, i] - gt_scaled[:, i], dim=-1) ** 2)).item() for i in indices]


def train():
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    os.makedirs(args['save_dir'], exist_ok=True)
    device = args['device']

    train_set = TrajectoryDataset(args['data_path'], mode='train', step=args['downsample_step'], scale=args['scale_factor'])
    val_set = TrajectoryDataset(args['data_path'], mode='val', step=args['downsample_step'], scale=args['scale_factor'])

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    model = TeacherNet(args).to(device)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr_teacher'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-5)

    best_5s = float('inf')
    patience_cnt = 0

    for ep in range(args['epochs']):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {ep + 1}/{args['epochs']}")

        for hist, fut, ego, neighbors in pbar:
            hist, fut, ego, neighbors = [x.to(device, non_blocking=True) for x in [hist, fut, ego, neighbors]]
            preds, _, _ = model(hist, ego, neighbors)
            loss = teacher_loss(preds, fut, ep, args['epochs'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['grad_clip'])
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(args['ema_decay']).add_(p.data, alpha=1.0 - args['ema_decay'])

        scheduler.step()

        ema_model.eval()
        val_rmses = []
        with torch.no_grad():
            for hist, fut, ego, neighbors in val_loader:
                hist, fut, ego, neighbors = [x.to(device, non_blocking=True) for x in [hist, fut, ego, neighbors]]
                preds, _, _ = ema_model(hist, ego, neighbors)
                val_rmses.append(compute_rmse(preds, fut, scale=args['scale_factor']))

        avg_rmse = np.mean(val_rmses, axis=0)
        print(f"[Epoch {ep + 1}] Loss: {total_loss / len(train_loader):.4f} | RMSE: "
              f"1s={avg_rmse[0]:.3f}, 2s={avg_rmse[1]:.3f}, 3s={avg_rmse[2]:.3f}, 4s={avg_rmse[3]:.3f}, 5s={avg_rmse[4]:.3f}")

        if avg_rmse[4] < best_5s:
            best_5s = avg_rmse[4]
            torch.save(ema_model.state_dict(), args['teacher_ckpt'])
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args['patience']:
                break


if __name__ == '__main__':
    train()
