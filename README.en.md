# Image Recognition for Droplet Sensors

Language option:
- [中文](README.md)

## Workflow

Extract frames from video -> Data labeling -> Model training/fine-tuning -> Inference on unknown videos

## Introduction

1. Using ResNet18 pre-trained model for image recognition

2. Adopted Full fine-tuning strategy due to significant differences between current images and pre-trained data

3. ~~Applied data augmentation (random rotation, brightness, etc.) due to small dataset size and low variance between samples~~(Resolved)

4. Developed a script to invoke the model, supporting frame extraction every "n" seconds as input

5. Output format as follows:

   <img width="433" height="274" alt="螢幕擷取畫面 2026-03-27 172553" src="https://github.com/user-attachments/assets/197997ea-e4c2-49f0-b212-c75aa72c9717" />


## Preliminary Results

1. With a sufficient number of images, data augmentation can be disabled.

2. Currently, the model's prediction results are stabilizing and highly accurate.

   <img width="800" height="600" alt="0327cm" src="https://github.com/user-attachments/assets/f4f05698-b899-4ea0-b648-1b6192686e8f" />


## Current Issues

1. ~~Dataset is too small and homogeneous, causing poor accuracy during inference on non-training videos (Gradually improving)~~(Resolved)

2. Insufficient computing power for direct video processing, necessitating frame extraction for image-based training

3. ~~Currently, embedding change timestamps directly into the video might affect model inference (to be verified)~~(Resolved)

## Future Prospects

1. Collect more diverse images to handle various scenarios ✅(That's enough)

2. Try larger models (e.g., ResNet50) if the dataset becomes sufficient

## Update Log

2026/2/5 : 

1. Split training and test sets, added evaluation metrics such as the confusion matrix, and added training images (162 total).

2026/3/27

1. Split/Refactor the program (Training folder)

2. Increased data volume (currently 379 photos)

3. Added ACC, Loss, and ROC curves during Training.

## References

Fine-Tuning a Pre-Trained ResNet-18 Model for Image Classification on Custom Dataset with PyTorch

https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385

直觀理解ResNet —簡介、 觀念及實作(Python Keras)

https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2

tif 2 jpg

https://github.com/santoshkaranam/convertTif2PngOrJpg
