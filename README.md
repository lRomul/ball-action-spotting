# Solution for SoccerNet Ball Action Spotting 2023 Challenge

![header](https://github.com/lRomul/ball-action-spotting/assets/11138870/df58e592-49c0-4904-8fb4-ce68ed143640)

This repo contains the solution for the [SoccerNet Ball Action Spotting 2023 Challenge](https://www.soccer-net.org/challenges/2023). 
The challenge goal is to develop an algorithm for spotting passes and drives occurring in videos of soccer matches. 
Unlike the [SoccerNet Action Spotting Challenge](https://www.soccer-net.org/tasks/action-spotting), the actions are much more densely allocated and should be predicted more accurately (with a 1-second precision). 
## Solution

Key points:
* Efficient model architecture for extracting information from videos data
* Multi-stage training (transfer learning, finetuning on long sequences)
* Fast video loading for training (GPU based, no need for preprocessing with extracting images)

### Model

The model architecture is a slow fusion approach using 2D convolutions in the early part and 3D convolutions in the late.
The architecture made one of the main contributions to the solution result. 
It raised the metric on test and challenge splits by ~0.15 mAP@1 (from 0.65 to 0.8) compared to the 2D CNN early fusion approach.

![model](https://github.com/lRomul/ball-action-spotting/assets/11138870/8e56bf90-d117-428f-b9bd-0927dab58107)

The model consumes sequences of grayscale frames. Neighboring frames are stacking as channels for input to the 2D convolutional encoder.
For example, if you take 15 frames and stack threes, you will get 5 input tensors with 3 channels for 2D convolutions. 
A single 2D encoder independently predicts those input tensors, producing visual features. 
3D encoder predicts visual features permuted to add temporal dimension. 
Concated temporal features from the 3D encoder pass through global pooling to compress the spatial dimensions. 
Then linear classifier predicts the presence of actions in the middle frame.

The core idea of the model is based on a concept from the 1st place solution of [DFL - Bundesliga Data Shootout Competition](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932).  

I choose the following model hyperparameters as a result of the experiments:
* Stack threes from 15 grayscale 1280x736 frames skipping every second frame in the original 25 FPS video (equivalent to 15 neighboring frames in 12.5 FPS, about 1.16 seconds window)
* EfficientNetV2 B0 as 2D encoder
* 4 inverted residual 3D blocks as 3D encoder (ported from 2D EfficientNet version)
* GeM as global pooling 
* Multilabel classification, positive labels in a 0.6 seconds window (15 frames in 25 FPS) around the action timestamp

You can find more details in the [model implementation](src/models/multidim_stacker.py) and [experiment configs](configs/ball_action).

### Training

I made several stages of training to obtain 86.47 mAP@1 on the challenge split (87.03 on the test): 
1. **Basic training.** The 2D encoder starts from ImageNet weights, and other parts start from scratch.
2. **Training on Action Spotting Challenge videos and classes.** Same weights as in p1.
3. **Transfer learning training.** 2D and 3D encoders start from p2 weights. Out-of-fold predictions from p1 were used for sampling. 
4. **Finetuning training on long sequences.** 2D and 3D encoders start from p3 weights. 2D encoder weights are frozen.

### Data loading

### Prediction and postprocessing

### Progress

You can see detailed progress of the solution development during the challenge in [spreadsheets](https://docs.google.com/spreadsheets/d/1mGnTdrVnhoQ8PJKNN539ZzhZxSowc4GpN9NdyDJlqYo/edit?usp=sharing).

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
