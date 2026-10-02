# Aerial Instance Segmentation

A CenterNet-style object detector for aerial imagery, built from scratch in PyTorch and trained on the iSAID dataset. The plan is to extend it with a mask head for instance segmentation.

**Status: in progress.** The detector network is built and tested. The data pipeline is in progress. The model has not been trained yet, so there are no results so far.

## Plan

1. Data pipeline: tile the large aerial images and build training targets from the annotations
2. Detector: train the network and evaluate it with box mAP
3. Mask head: add per-object masks, turning detection into instance segmentation
4. Pretrained backbone: compare my encoder against a frozen pretrained one

## Model

- U-Net style encoder (four downsampling blocks), with a decoder that stops at a quarter of the input resolution
- Three heads: a heatmap of object centres (one channel per class), object width and height, and a sub-pixel offset for each centre
- 30.8 million parameters
- The two widest decoder blocks run in fp32 under mixed-precision training, to prevent an fp16 overflow I found in an earlier U-Net project

## Dataset

iSAID: 15 object classes across 2,806 aerial images (1,411 train, 458 val, 937 test). The images come from DOTA v1.0. The dataset is not included in this repo, as it is released for non-commercial research use. I downloaded the annotations and images from the Hugging Face datasets isaaccorley/isaid and isaaccorley/dota.

## Repository layout

- src/model.py: the detector network
- notebooks/01_model_tests.ipynb: shape and sanity checks for the network
- notebooks/02_data_exploration.ipynb: downloading and inspecting the dataset

## References

- Zamir et al., "iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images", CVPR Workshops 2019
- Zhou et al., "Objects as Points" (CenterNet), 2019

Built as a learning project, with AI assistance for guidance and setup.