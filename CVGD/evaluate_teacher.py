import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from loader import TrajectoryDataset
from models.teacher_model import TeacherNet
from train_teacher import compute_rmse


class EvaluateTeacher:
    def __init__(self, args):
        self.args = args
        self.device = args['device']

    def run(self):
        val_set = TrajectoryDataset(
            self.args['data_path'],
            mode='val',
            step=self.args['downsample_step'],
            scale=self.args['scale_factor']
        )
        val_loader = DataLoader(val_set, batch_size=self.args['batch_size'], shuffle=False, num_workers=4)

        net = TeacherNet(self.args).to(self.device)
        net.load_state_dict(torch.load(self.args['teacher_ckpt'], map_location=self.device))
        net.eval()

        all_rmses = []
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for hist, fut, ego, neighbors in val_loader:
                hist, fut, ego, neighbors = [x.to(self.device) for x in [hist, fut, ego, neighbors]]

                t0 = time.time()
                preds, _, _ = net(hist, ego, neighbors)
                total_time += (time.time() - t0)
                total_samples += hist.shape[0]

                all_rmses.append(compute_rmse(preds, fut, scale=self.args['scale_factor']))

        avg_rmse = np.mean(all_rmses, axis=0)
        print("==========================================")
        print("Teacher Model Evaluation Results (RMSE in meters):")
        for sec, val in zip([1, 2, 3, 4, 5], avg_rmse):
            print(f"  Horizon {sec}s: {val:.4f} m")
        print(f"Average Inference Time per Sample: {(total_time / total_samples) * 1000:.3f} ms")
        print("==========================================")


if __name__ == '__main__':
    evaluator = EvaluateTeacher(args)
    evaluator.run()
