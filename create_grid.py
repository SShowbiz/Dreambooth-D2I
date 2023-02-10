import argparse
from tqdm import tqdm
from PIL import Image
import os

SIZE = 384

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", default="results_seed0")
args = parser.parse_args()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

if __name__ == "__main__":
    source = os.listdir(args.source_dir)

    grid = Image.new('RGB', size=(3*SIZE, len(source)*SIZE))
    for idx, image_dir in enumerate(tqdm(source)):
        depthmap_img = Image.open(os.path.join(args.source_dir, image_dir, 'depthmap.png'))
        input_img = Image.open(os.path.join(args.source_dir, image_dir, 'input.png'))
        output_img = Image.open(os.path.join(args.source_dir, image_dir, 'output.png'))

        images = [input_img, depthmap_img, output_img]
        for i, img in enumerate(images):
            grid.paste(img, box=(i*SIZE, idx*SIZE))

    grid.save('grid.png')