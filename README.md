# Solution for SoccerNet Ball Action Spotting 2023 Challenge

![header](https://github.com/lRomul/ball-action-spotting/assets/11138870/df58e592-49c0-4904-8fb4-ce68ed143640)

This repo contains the source code of the solution for [SoccerNet Ball Action Spotting 2023 Challenge](https://www.soccer-net.org/challenges/2023). 
The goal of the challenge is to develop an algorithm for spotting passes and drives occurring in videos of soccer matches. 
Unlike the [SoccerNet Action Spotting Challenge](https://www.soccer-net.org/tasks/action-spotting), the actions are much more densely allocated and should be predicted more accurately (with a 1-second precision). 

## Solution

Key points:
* Efficient model architecture for extracting information from videos data
* Multi-stage training (transfer learning, finetuning on long sequences)
* Fast video loading for training (GPU based, no need for preprocessing with extracting images)

### Model

The model architecture is a slow fusion approach that uses 2D convolutions in the early part and 3D convolutions in the late.
The architecture made one of the main contributions to the solution result. 
It raised the metric on test and challenge splits by ~0.15 mAP@1 (from 0.65 to 0.8) compared to the 2D CNN early fusion approach.

<img src="https://github.com/lRomul/ball-action-spotting/assets/11138870/8e56bf90-d117-428f-b9bd-0927dab58107"  width="80%">

The model consumes sequences of grayscale frames. Neighboring frames are stacking as channels for input to the 2D convolutional encoder.
For example, if there are 15 frames and triples are stacked, then we get 5 input tensors with 3 channels for 2D convolutions. 
A single 2D encoder independently predicts those input tensors, thereby producing visual features. 
3D encoder predicts visual features permuted to add temporal dimension. 
Concated temporal features from the 3D encoder pass through global pooling to compress the spatial dimensions. 
Then linear classifier predicts the presence of actions in the central frame.

The core idea of the model is based on a concept from the 1st place solution of [DFL - Bundesliga Data Shootout Competition](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932).  

### Training

### Data loading

### Progress

Detailed progress of the solution development during the challenge can be seen in spreadsheets:

## Quick setup and start

### Requirements

* Linux (tested on Ubuntu 20.04 and 22.04)
* NVIDIA GPU (pipeline tuned for RTX 3090)
* NVIDIA Drivers >= 520, CUDA >= 11.8
* [Docker](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Preparations

* Clone the repo.
    ```bash
    git clone git@github.com:lRomul/ball-action-spotting.git
    cd ball-action-spotting
    ```

* 

### Run
