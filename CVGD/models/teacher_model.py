import math
import torch
import torch.nn as nn


class ContinuousAVS(nn.Module):
    def __init__(self, ego_dim=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, ego, neighbors):
        valid_mask = (neighbors != 100.0).any(dim=-1, keepdim=True)
        neighbors = torch.where(valid_mask, neighbors, torch.zeros_like(neighbors))

        p = self.mlp(ego)
        angle = torch.sigmoid(p[:, 0]) * math.pi
        radius = torch.sigmoid(p[:, 1]) * 20.0
        sharp = torch.sigmoid(p[:, 2]) * 10.0
        ellipse = torch.sigmoid(p[:, 3]) * 2.0

        dx, dy = neighbors[:, :, 0], neighbors[:, :, 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        theta = torch.atan2(dy, dx)
        diff = torch.abs(theta - angle.unsqueeze(1))

        w = torch.exp(-sharp.unsqueeze(1) * diff) \
            - torch.exp(-dist / (radius.unsqueeze(1) + 1e-6)) \
            - torch.clamp(1.0 + ellipse.unsqueeze(1) * torch.cos(2 * diff), 0.5, 2.0)

        w = w * valid_mask.squeeze(-1).float()
        return w.unsqueeze(-1), neighbors


class LATGDecoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2 + 1 + latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


class TeacherNet(nn.Module):
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
