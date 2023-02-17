import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default=None, help="source image directory")
parser.add_argument('--output_path', default=None, help="edge image directory")
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    images = os.listdir(args.input_path)
    for image_name in images:
        image_path = os.path.join(args.input_path, image_name)
        save_path = os.path.join(args.output_path, image_name)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray_image, 50, 150)
        cv2.imwrite(save_path, edge)