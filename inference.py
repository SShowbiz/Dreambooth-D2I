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
parser.add_argument("--depthmap_dir", default=None, type=str, help="depthmap directory")
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
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        self.to_pil = transforms.ToPILImage()
        self.positive_prompt = args.positive_prompt
        self.negative_prompt = args.negative_prompt
        self.depthmap_dir = args.depthmap_dir
        self.source_dir = args.source_dir
        self.save_dir = args.save_dir

    def preprocess(self, image):
        image = self.transform(image)[None, :, :, :].to("cuda")
        return image

    def create_depthmap(self, image):
        depthmap = self.pipeline.depth_estimator(image).predicted_depth
        return depthmap

    def save_images(self, image_name, input_img, depthmap_img, output_img, output_only=False):
        save_dir = self.save_dir if output_only else os.path.join(self.save_dir, image_name) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if output_only:
            output_img.save(os.path.join(save_dir, f"{image_name}.png"))
        else:
            output_img.save(os.path.join(save_dir, "output.png"))
            self.to_pil(input_img[0]).save(os.path.join(save_dir, "input.png"))
            self.to_pil(depthmap_img).save(os.path.join(save_dir, "depthmap.png"))
        

    def __call__(self, output_only=False):
        images = os.listdir(self.source_dir)
        for image in images:
            image_path = os.path.join(self.source_dir, image)
            image_name = image.split(".")[0]

            if self.depthmap_dir is not None:
                depthmap_path = os.path.join(self.depthmap_dir, image)
                depthmap = self.preprocess(Image.open(depthmap_path))
                image = self.preprocess(Image.open(image_path))
            else:
                image = self.preprocess(Image.open(image_path))
                depthmap = self.create_depthmap(image)

            # remove alpha channel if exists
            N, C, *_ = depth_map.shape
            if N == C == 1:
                depth_map = depth_map.squeeze(0)

            generator = torch.Generator(device="cuda")
            generator.manual_seed(args.seed)
            
            image = transforms.ToPILImage()(image[0])
            output = self.pipeline(
                prompt=self.positive_prompt,
                image=image,
                negative_prompt=self.negative_prompt,
                strength=0.8,
                num_inference_steps=200,
                guidance_scale=7,
                depth_map=depthmap,
                generator=generator,
            )[0][0]
                
            self.save_images(
                image_name=image_name,
                input_img=image,
                depthmap_img=depthmap,
                output_img=output,
                output_only=output_only
            )

if __name__ == "__main__":
    dreambooth = DreamBooth(args)
    dreambooth(output_only=args.output_only)
    