# Dreambooth-Depth2Img

This repository is for fine-tuning Dreambooth-Depth2Img which is originally suggested in [stabilityai/stable-diffusion-2-depth](https://huggingface.co/stabilityai/stable-diffusion-2-depth).

## Depthmap Generation

### Monocular Depth Estimation

To use 512 dimension depth map estimation, we use MiDas pretrained model.

```shell
$ cd depth/monodepth/weights
$ wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

Above command will save pretrained MiDas model into appropriate folder.

```shell
$ cd depth/monodepth
$ python estimate_depth.py --input_path ${SOURCE_DIR} --output_path ${DEPTH_DIR} --grayscale
```

Above command will inference depthmap of images in ${SOURCE_DIR}.

### Edge Estimation

To add edge information to depthmap, we can use Canny Edge Detector.

```shell
$ cd depth/depth/edge
$ python estimate_edge.py --input_path ${SOURCE_DIR} --output_path ${EDGE_DIR}
```

### Face Parsing

To add face parsing information to depthmap and generate final depth image, you can use below command.

```shell
$ python edit_depth.py --input_path ${SOURCE_DIR} --depth_path ${DEPTH_DIR} --edge_path ${EDGE_DIR} --output_path ${FINAL_DEPTH_DIR}
```

Above command uses original depth and edge images and apply it to final depthmap.
