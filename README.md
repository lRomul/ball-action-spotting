# Solution for SoccerNet Ball Action Spotting 2023 Challenge

This repo contains the source code of the solution for [SoccerNet Ball Action Spotting 2023 Challenge](https://www.soccer-net.org/challenges/2023). 
The goal of the challenge is to develop an algorithm for spotting passes and drives occurring in videos of soccer matches. 
Unlike the [SoccerNet Action Spotting Challenge](https://www.soccer-net.org/tasks/action-spotting), the actions are much more dense allocated, and should be predicted more accurately (with a 1-second precision). 

## Solution

Key points:
* Efficient model architecture for extracting information from videos data
* Multi-stage training (transfer learning, finetuning on long sequences)
* Fast video loading for training (GPU based, no need preprocessing with extracting images)

### Model

![Model](https://github.com/lRomul/ball-action-spotting/assets/11138870/f75aaaf0-664a-466c-8193-e9bba7cb0926)

### Training

### Data loading

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
