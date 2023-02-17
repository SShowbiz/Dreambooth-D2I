from models.generator import Generator
import torch
from utils import convert_img, mixing_noise
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Process")
parser.add_argument("--num", default=10000, type=int, help="")
parser.add_argument("--model_path", default="pretrain_models/stylegan2-ffhq-config-f.pt", type=str, help="")
parser.add_argument("--output_path", default="generated_ffhq", type=str, help="")
args = parser.parse_args()


class Process:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_dim = 512
        self.size = 1024
        self.n_mlp = 8
        self.channel_multiplier = 2
        self.netGs = Generator(
            size=self.size,
            style_dim=self.latent_dim,
            n_mlp=self.n_mlp,
            channel_multiplier=self.channel_multiplier,
        ).to(self.device)

        self.loadparams(args.model_path)
        self.num_gen = args.num
        self.truncation = 0.4

        with torch.no_grad():
            self.mean_latent = self.netGs.mean_latent(4096)

        self.netGs.eval()

    def __call__(self, save_base):
        os.makedirs(save_base, exist_ok=True)
        steps = 0
        for i in range(self.num_gen):
            ffhq = self.run_sigle()
            for f in ffhq:
                cv2.imwrite(os.path.join(save_base, "%06d.png" % steps), f)
                steps += 1
                print("\r have done %06d" % steps, end="", flush=True)


    def run_sigle(self):
        noise = mixing_noise(batch=1, latent_dim=self.latent_dim, prob=0.9, device=self.device)
        with torch.no_grad():
            fake_s, _ = self.netGs(
                noise, truncation=self.truncation, truncation_latent=self.mean_latent, return_latents=True
            )
            fake = convert_img(fake_s, unit=True).permute(0, 2, 3, 1)
            return fake.cpu().numpy()[..., ::-1]

    def loadparams(self, path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netGs.load_state_dict(ckpt["g_ema"], strict=False)

if __name__ == "__main__":
    model = Process(args)
    model(args.output_path)
