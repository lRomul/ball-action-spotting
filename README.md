# Solution for SoccerNet Ball Action Spotting 2023 Challenge

![header](https://github.com/lRomul/ball-action-spotting/assets/11138870/df58e592-49c0-4904-8fb4-ce68ed143640)

This repo contains the solution for the [SoccerNet Ball Action Spotting 2023 Challenge](https://www.soccer-net.org/challenges/2023). 
The challenge goal is to develop an algorithm for spotting passes and drives occurring in videos of soccer matches. 
Unlike the [SoccerNet Action Spotting Challenge](https://www.soccer-net.org/tasks/action-spotting), the actions are much more densely allocated and should be predicted more accurately (with a 1-second precision).

## Solution

Key points:
* Efficient model architecture for extracting information from videos data
* Multi-stage training (transfer learning, fine-tuning with long sequences)
* Fast video loading for training (GPU based, no need for preprocessing with extracting images)

### Model

The model architecture is a slow fusion approach using 2D convolutions in the early part and 3D convolutions in the late.
The architecture made one of the main contributions to the solution result. 
It raised the metric on test and challenge sets by ~15% mAP@1 (from 65 to 8) compared to the 2D CNN early fusion approach.

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

I made several stages of training to obtain 86.47% mAP@1 on the challenge set (87.03% on the test): 
1. **Basic training ([config](configs/ball_action/sampling_weights_001.py)).** The 2D encoder starts from ImageNet weights, and other parts start from scratch.
2. **Training on Action Spotting Challenge dataset ([config](configs/action/action_sampling_weights_002.py)).** Same weights as in 1.
3. **Transfer learning ([config](configs/ball_action/ball_tuning_001.py)).** 2D and 3D encoders start from 2 weights. Out-of-fold predictions from 1 were used for data sampling (more details later).
4. **Fine-tuning with long sequences ([config](configs/ball_action/ball_finetune_long_004.py)).** 2D and 3D encoders start from 3 weights. 2D encoder weights are frozen.

#### Basic training

In this challenge, I used 7-fold cross-validation to tune the training pipeline more precisely. 
Each labeled game from the dataset is a different fold. 

In short, the resulting training pipeline:
* Learning rate warmup first 6 epochs from 0 to 3e-4, cosine annealing last 30 epochs to 3e-6
* Batch size 4, one training epoch 6000 samples 
* Optimizer AdamW with weight decay 0.01
* Focal loss with gamma 1.2
* Model EMA with decay 0.999
* Initial weights for 2D encoder ImageNet pretrained
* Model hyperparameters listed above in the model part

Worth writing about sampling techniques during training, which significantly impacts its results. 
For basic training, was used simple but well work sampling algorithm. 
For each training sample, randomly take video index by a uniform distribution. 
Then randomly choose a frame index by the following distribution. 
Large values are placed around event labels in a window of 9 frames. 
Values are normalized so that the sum of probabilities around actions equals the sum around non-action frames.
I tried different ratios, but an equal chance to show empty and event frame worked best. 
I will introduce a more advanced sampling scheme in part about transfer learning.

The models from this training have 79.06% on CV (cross-validation) and 84.26% mAP@1 on the test set (the metric on test split was calculated by the out-of-fold predictions for two folds which include test games). 
I didn't evaluate these models for the challenge set.

#### Training on Action Spotting Challenge dataset

I built a similar pipeline for videos and classes from Action Spotting Challenge to get good initial weights for the next experiment. 

Briefly, here are the changes from the previous:
* 377 games in training, 18 in validation split
* 15 classes (cards class was merged because the model consumes grayscale frames) 
* 4 warmup epochs and 20 training epochs, one epoch is 36000 samples
* Weight the frame sampling by the effective number of class samples 

#### Transfer learning

This training uses the results of the previous two. 
The second one gives excellent initial weights for 2D and 3D encoders. It provides a significant boost (~2% mAP@1 on test and CV). 
That is understandable because the same models were trained on many games with similar input frames and some similar actions.

Basic training gives out-of-fold predictions that I use for sampling this way:

![sampling](https://github.com/lRomul/ball-action-spotting/assets/11138870/737c37f1-4f03-4344-a670-64f20c6972fc)

Take the element-wise maximum between the sampling distribution (introduced above) and predictions, then normalize again to equal probability sums between empty and action frames.
The intuition is that there are some hard negative examples in the dataset. Due to many negative samples, such hard examples are rarely sampled during training. 
We can make something like hard negative mining/sampling with the technique.

Other minor changes compared to basic training:
* 7 warmup epochs and 35 training epochs
* Focal loss with gamma 1.2 and alpha 0.4

Models achieve 81.04% on CV, 86.51% on the test, and 86.35% mAP@1 on the challenge set.

#### Fine-tuning with long sequences

Before, I trained all models on relatively short clips (15 frames in 12.5 FPS, 1.16 seconds) to fits the VRAM of a single GPU and to decrease training time. Models obtained good 2D and 3D encoder weights in the previous experiment. So I tried to freeze 2D weights and fine-tune 3D weights on long sequences to provide more context. 

Changes compared to transfer learning experiment:
* 33 frames in 12.5 FPS (2.6 seconds)
* LR warmup first 2 epochs from 0 to 1e-3, cosine annealing last 7 epochs to 5e-05
* SGD with Nesterov momentum 0.9

Models scored 80.49% on CV, 87.04% on the test, and 86.47% mAP@1 on the challenge set. The score is lesser on cross-validation, but it's my best submission on the test and the challenge set ¯\_(ツ)_/¯

#### Prediction and postprocessing

#### Training and prediction accelerations

#### Progress

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
