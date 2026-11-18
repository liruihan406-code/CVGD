import torch
import torch.nn as nn
from models.teacher_model import ContinuousAVS, LATGDecoder


class StudentNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args['hidden_dim']
        self.latent_dim = args['latent_dim']
        self.num_modes = args['num_modes']
        self.out_length = args['out_length']

        self.encoder = nn.LSTM(2, self.hidden_dim, batch_first=True)
        self.avs = ContinuousAVS(ego_dim=args['ego_dim'])
        self.pool_proj = nn.Linear(2, self.hidden_dim)
        self.decoder = LATGDecoder(self.hidden_dim, self.latent_dim)
        self.mode_head = nn.Linear(self.hidden_dim, self.num_modes)

        # Projection adapter for feature-level distillation
        self.feat_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def encode(self, hist, ego, neighbors):
        self.encoder.flatten_parameters()
        _, (h, _) = self.encoder(hist)
        feat = h.squeeze(0)

        w, clean_neighbors = self.avs(ego, neighbors)
        sum_w = torch.sum(w, dim=1) + 1e-6
        pooled = torch.sum(w * clean_neighbors, dim=1) / sum_w

        return feat + self.pool_proj(pooled)

    def forward(self, hist, ego, neighbors, z=None):
        feat = self.encode(hist, ego, neighbors)
        logit = self.mode_head(feat)

        B = hist.shape[0]
        feat_expand = feat.repeat_interleave(self.num_modes, dim=0)
        curr = hist[:, -1, :].repeat_interleave(self.num_modes, dim=0)
        v = (hist[:, -1, :] - hist[:, -2, :]).repeat_interleave(self.num_modes, dim=0)

        if z is None:
            z = torch.randn(B * self.num_modes, self.latent_dim, device=hist.device)

        traj = []
        for s in range(self.out_length):
            t = torch.full((B * self.num_modes, 1), s / self.out_length, device=hist.device)
            dv = self.decoder(torch.cat([feat_expand, curr, t, z], dim=-1))
            v = v + dv * 0.1
            curr = curr + v
            traj.append(curr.unsqueeze(1))

        samples = torch.cat(traj, dim=1).view(B, self.num_modes, self.out_length, 2)
        return samples, feat, logit
