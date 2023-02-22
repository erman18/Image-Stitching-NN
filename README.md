# Deep Learning Model for Image Stitching

Credit: Implementation based on the [Image Super Resolution Repos](https://github.com/titu1994/Image-Super-Resolution)

# Image stitching

## Data Preparation

Generate the training dataset from real images

```bash
./prepare_data.sh
```

## Traning

```bash
python train.py --model UnDDStitch --batch_size=16 --nb_epochs=20 --load_weights 1 --save_model_img 0 --supervised 1

```

## Inference

```
./main_stitching.sh
```
