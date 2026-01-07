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

## References

Fine-Tuning a Pre-Trained ResNet-18 Model for Image Classification on Custom Dataset with PyTorch

https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385

直觀理解ResNet —簡介、 觀念及實作(Python Keras)

https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2
