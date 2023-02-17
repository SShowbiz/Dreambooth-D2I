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


## Fine Tuning

You can fine tune the model using below command. You should prepare for the few shot images and the corresponding prompt.

```shell
$ python train_dreambooth.py  \
--mixed_precision "fp16" \
--pretrained_model_name_or_path stabilityai/stable-diffusion-2-depth  \
--pretrained_txt2img_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
--train_text_encoder \
--instance_data_dir ${FEWSHOT_IMAGES_DIR} \
--output_dir ${CHECKPOINT_SAVE_DIR} \
--instance_prompt ${FEWSHOT_OBJECT_PROMPT} \
--resolution 512 \
--train_batch_size 4 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-6 \
--lr_scheduler "constant" \
--lr_warmup_steps 0 \
--max_train_steps 500 \
--use_8bit_adam
```

Above command will generated overfitted, bias text encoder embedding space since it doesn't have prior preservation term.
Use below command for use prior preservation.

```shell
python train_dreambooth.py  \
--mixed_precision="fp16" \
--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth  \
--pretrained_txt2img_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--train_text_encoder  \
--instance_data_dir ${FEWSHOT_IMAGES_DIR} \
--class_data_dir ${PRIOR_IMAGES_DIR} \
--output_dir ${CHECKPOINT_SAVE_DIR} \
--with_prior_preservation \
--prior_loss_weight 1.0 \
--instance_prompt ${FEWSHOT_OBJECT_PROMPT} \
--class_prompt ${PRIOR_IMAGE_PROMPT} \
--resolution 512 \
--train_batch_size 4 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-6 \
--lr_scheduler "constant" \
--lr_warmup_steps 0 \
--num_class_images 200 \
--max_train_steps 300 \
--use_8bit_adam
```

## Inference

By using fine tuned model, you can inference images with below command.

```shell
$ python inference.py  \
--ckpt ${CHECKPOINT_DIR} \
--source_dir ${IMAGES_FOR_INFERENCE} \
--save_dir ${IMAGE_SAVE_DIR} \
--seed 0 \
--positive_prompt ${POSITIVE_PROMPT} \
--negative_prompt ${NEGATIVE_PROMPT} \
--output_only
```
