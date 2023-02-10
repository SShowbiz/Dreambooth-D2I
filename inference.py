import torch
from torchvision import transforms
from diffusers import StableDiffusionDepth2ImgPipeline
import os
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/output", type=str, help="image source directory")
parser.add_argument("--source_dir", default="ffhq", type=str, help="image source directory")
parser.add_argument("--save_dir", default="results", type=str, help="image save directory")
parser.add_argument("--seed", default=0, type=int, help="seed for random generator")
parser.add_argument(
    "--positive_prompt",
    default="a portrait of limjukyung, {{best quality}}, {{masterpiece}}, {highres}, solo, sharp focus, extremely detailed character",
    type=str,
    help="positive prompt",
)
parser.add_argument(
    "--negative_prompt",
    default="NSFW, poorly drawn, text, text balloon, bad anatomy, bad proportions, bad face, bad hands, bad body, worst quality, low quality, normal quality, blurry, artifact",
)
parser.add_argument("--output_only", action="store_true")
args = parser.parse_args()


class DreamBooth:
    def __init__(self, args) -> None:
        self.pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(args.ckpt).to(
            "cuda"
        )
        self.transform = transforms.Compose(
            [transforms.Resize((384, 384)), transforms.ToTensor()]
        )
        self.to_pil = transforms.ToPILImage()
        self.positive_prompt = args.positive_prompt
        self.negative_prompt = args.negative_prompt
        self.source_dir = args.source_dir
        self.save_dir = args.save_dir
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(args.seed)

    def preprocess(self, image):
        image = self.transform(image)[None, :, :, :].to("cuda")
        return image

    def create_depthmap(self, image):
        depthmap = self.pipeline.depth_estimator(image).predicted_depth
        depth_min = torch.amin(depthmap, dim=[0, 1, 2], keepdim=True)
        depth_max = torch.amax(depthmap, dim=[0, 1, 2], keepdim=True)
        depthmap = 2.0 * (depthmap - depth_min) / (depth_max - depth_min) - 1.0
        depthmap = depthmap[0, :, :]

        return depthmap

    def save_images(self, subdir, input_img, depthmap_img, output_img, output_only=False):
        save_dir = self.save_dir if output_only else os.path.join(self.save_dir, subdir) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        output_img.save(os.path.join(save_dir, "output.png"))
        if not output_only:
            self.to_pil(input_img[0]).save(os.path.join(save_dir, "input.png"))
            self.to_pil(depthmap_img).save(os.path.join(save_dir, "depthmap.png"))
        

    def __call__(self, output_only=False):
        images = os.listdir(self.source_dir)
        for image in images:
            image_path = os.path.join(self.source_dir, image)
            image_name = image.split(".")[0]
            image = self.preprocess(Image.open(image_path))

            depthmap = self.create_depthmap(image)
            output = self.pipeline(
                prompt=self.positive_prompt,
                image=image,
                negative_prompt=self.negative_prompt,
                strength=0.8,
                num_inference_steps=200,
                guidance_scale=7,
                generator=self.generator,
            )[0][0]
                
            self.save_images(
                subdir=image_name,
                input_img=image,
                depthmap_img=depthmap,
                output_img=output,
                output_only=output_only
            )

if __name__ == "__main__":
    dreambooth = DreamBooth(args)
    dreambooth(output_only=args.output_only)
    