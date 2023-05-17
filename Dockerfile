FROM osaiai/dokai:23.05-vpf

RUN pip3 install --no-cache-dir \
    SoccerNet==0.1.51 \
    rosny==0.0.6

ENV OPENCV_FFMPEG_CAPTURE_OPTIONS "video_codec;hevc"
