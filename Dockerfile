FROM osaiai/dokai:22.11-pytorch

RUN apt-get update && \
    apt-get -y install \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev \
    libnvidia-encode-520 \
    libnvidia-decode-520

RUN git clone --branch=master --single-branch https://github.com/NVIDIA/VideoProcessingFramework && \
    cd VideoProcessingFramework \
    git checkout ec8caf75c341b9f4a125db59676c6742c8b666fd && \
    pip3 install . && \
    cd src/PytorchNvCodec && pip3 install . && \
    cd ../../.. && rm -rf VideoProcessingFramework

RUN pip3 install --no-cache-dir \
    SoccerNet==0.1.46 \
    rosny==0.0.6 \
    kornia==0.6.10
