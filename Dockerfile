FROM osaiai/dokai:22.11-pytorch

ENV OPENCV_FFMPEG_CAPTURE_OPTIONS "video_codec;hevc"

RUN pip3 install SoccerNet==0.1.46
