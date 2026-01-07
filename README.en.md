# Image Recognition for Droplet Sensors

Language option:
- [中文](README.md)

## Workflow

Extract frames from video -> Data labeling -> Model training/fine-tuning -> Inference on unknown videos

## Introduction

1. Using ResNet18 pre-trained model for image recognition

2. Adopted Full fine-tuning strategy due to significant differences between current images and pre-trained data

3. Applied data augmentation (random rotation, brightness, etc.) due to small dataset size and low variance between samples

4. Developed a script to invoke the model, supporting frame extraction every "n" seconds as input

5. Output format as follows:

<img width="406" height="229" alt="image" src="https://github.com/user-attachments/assets/bc588f7a-b8b3-4ac4-b488-924276a13589" />

## Preliminary Results

1. Training loss and accuracy are satisfactory; the model converges

2. Can identify the specific seconds where changes occur in the training videos

## Current Issues

1. Dataset is too small and homogeneous, causing poor accuracy during inference on non-training videos

2. Insufficient computing power for direct video processing, necessitating frame extraction for image-based training

## Future Prospects

1. Collect more diverse images to handle various scenarios

2. Try larger models (e.g., ResNet50) if the dataset becomes sufficient

3. Quantification issues
