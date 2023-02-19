from model import BiSeNet

import torch
import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/79999_iter.pth", type=str, help="checkpoint path")
parser.add_argument("--input_path", default="ffhq", type=str, help="datasets for original image")
parser.add_argument("--depth_path", default="ffhq_depth", type=str, help="datasets for depthmap to edit")
parser.add_argument("--edge_path", default="ffhq_edge_canny", type=str, help="datasets for edge of original image")
parser.add_argument("--output_path", default="results18", type=str, help="directory for save edited depthmap")
args = parser.parse_args()

VAL = 5
EG_EYE_VAL = 5
EG_MOUTH_VAL = 2
EG_HAIR_VAL = 2
EYE_VAL = 5
HAIR_VAL = 10
LIP_VAL = 10
EYEBROW_VAL= 5
MOUTH_VAL = 5
NOSE_VAL = 5
THRESHOLD = 120
EG_TRHESHOLD = 200

class ProcessDepthMap:
    def __init__(self, args):
        self.ckpt = args.ckpt
        self.input_path = args.input_path
        self.depth_path = args.depth_path
        self.output_path = args.output_path
        self.edge_path = args.edge_path
    
    def scale_contour(self, cnt, scale):
        M = cv2.moments(cnt)

        if M['m00'] == 0:
            cx, cy = np.mean(cnt, axis=0)
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        
        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled
    
    def vis_parsing_maps(self, im, dm, eg, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):        
        part_colors = [0, 0, EYEBROW_VAL, EYEBROW_VAL, -EYE_VAL, -EYE_VAL, -EYE_VAL, 0, 0, 0, NOSE_VAL, -MOUTH_VAL, LIP_VAL, LIP_VAL, 0, 0, 0, -HAIR_VAL]
        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        
        part2idx = {part:idx for idx, part in enumerate(atts)}
        editing_region = ['nose', 'mouth', 'l_brow', 'r_brow', 'u_lip', 'l_lip', 'l_eye', 'r_eye', 'hair']

        im_gray = im.convert("L")
        im_gray = np.array(im_gray)
        im = np.array(im)
        np_dm = np.clip(np.array(dm) / 255.0, 0, 255)

        eg = eg.convert("L")
        eg = np.array(eg)

        eg_eye = eg.copy()
        eg_hair = eg.copy()
        eg_mouth = eg.copy()

        eg_hair[eg_hair < 150] = 0

        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))

        for region in editing_region:
            idx = part2idx[region] + 1
            index = np.where(vis_parsing_anno == idx)
            vis_parsing_anno_color[index[0], index[1]] = part_colors[idx]

            if region in ['l_eye', 'r_eye']:
                vis_parsing_anno_eye = np.copy(vis_parsing_anno)

                vis_parsing_anno_eye[vis_parsing_anno_eye != idx] = 0
                vis_parsing_anno_eye[vis_parsing_anno_eye == idx] = 1
                # vis_parsing_anno_eye_contour = np.zeros_like(vis_parsing_anno_eye)
                # vis_parsing_anno_eg_contour = np.zeros_like(vis_parsing_anno_eye)

                kernel = np.ones((3, 3), np.uint8)
                vis_parsing_anno_eye = cv2.dilate(vis_parsing_anno_eye, kernel, iterations=10)

                index = np.where((vis_parsing_anno_eye == 1) & (eg_eye > 0))
                vis_parsing_anno_color[index[0], index[1]] -= EG_EYE_VAL

                # contours, _ = cv2.findContours(vis_parsing_anno_eye, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # if len(contours) > 0:
                #     selected_contour = contours[0]
                #     for contour in contours:
                #         num, _, __ = contour.shape
                #         if num > selected_contour.shape[0]:
                #             selected_contour = contour

                #     selected_contour = selected_contour.squeeze(1)
                    
                #     vis_parsing_anno_eg_contour = cv2.fillPoly(vis_parsing_anno_eg_contour, pts=[self.scale_contour(selected_contour, 2.5)], color=[255])
                #     vis_parsing_anno_eye_contour = cv2.fillPoly(vis_parsing_anno_eye_contour, pts=[self.scale_contour(selected_contour, 1.5)], color=[255])
                #     index_eg = np.where((vis_parsing_anno_eg_contour == 255) & (eg_eye > 0))
                #     vis_parsing_anno_color[index_eg[0], index_eg[1]] -= EG_EYE_VAL
                   

                # index = np.where(vis_parsing_anno_eye_contour == 255)
                # vis_parsing_anno_color[index[0], index[1]] += part_colors[idx]

        
        for region in ['l_eye', 'r_eye']:
            idx = part2idx[region] + 1

            index = np.where((vis_parsing_anno == idx) & (im_gray < THRESHOLD))
            vis_parsing_anno_color[index[0], index[1]] -= EYE_VAL

        # for region in ['hair']:
        #     idx = part2idx[region] + 1
        #     index = np.where((vis_parsing_anno == idx) & (eg_hair > 0))
        #     vis_parsing_anno_color[index[0], index[1]] -= EG_HAIR_VAL
        
        for region in ['mouth', 'u_lip', 'l_lip']:
            idx = part2idx[region] + 1

            vis_parsing_anno_mouth = np.copy(vis_parsing_anno)
            vis_parsing_anno_mouth[vis_parsing_anno_mouth != idx] = 0
            vis_parsing_anno_mouth[vis_parsing_anno_mouth == idx] = 1

            kernel = np.ones((3, 3), np.uint8)
            vis_parsing_anno_mouth = cv2.dilate(vis_parsing_anno_mouth, kernel, iterations=15)

            index = np.where((vis_parsing_anno_mouth == 1) & (eg_mouth > 0))
            vis_parsing_anno_color[index[0], index[1]] -= EG_MOUTH_VAL

        np_vis = np_dm + vis_parsing_anno_color
        min_val = np.min(np_vis)
        max_val = np.max(np_vis)
        np_vis = (np_vis - min_val) / (max_val - min_val) * 255.0

        pil_image = Image.fromarray(np_vis.astype(np.uint8))
        pil_image.save(save_path)

    def evaluate(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        # save_pth = osp.join('res/cp', cp)
        net.load_state_dict(torch.load(self.ckpt))
        net.eval()

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            for image_path in os.listdir(self.input_path):
                img = Image.open(osp.join(self.input_path, image_path))
                image = img.resize((512, 512), Image.BICUBIC)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()

                depthmap = Image.open(osp.join(self.depth_path, image_path))
                depthmap = depthmap.resize((512, 512), Image.BICUBIC)

                edge = Image.open(osp.join(self.edge_path, image_path))
                edge = edge.resize((512, 512), Image.BICUBIC)

                out = net(img)[0]

                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                print(np.unique(parsing))

                self.vis_parsing_maps(image, depthmap, edge, parsing, stride=1, save_im=True, save_path=osp.join(self.output_path, image_path))


if __name__ == "__main__":
    process = ProcessDepthMap(args)
    process.evaluate()
