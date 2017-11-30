FROM opencv2-cuda8 
RUN apt-get update && \
    apt-get install -y git qtbase5-dev

RUN mkdir -p /src
RUN mkdir -p /video
RUN mkdir -p /output

VOLUME /video
VOLUME /output

WORKDIR /src
ADD . /src/gpu_flow
RUN mkdir gpu_flow_build && cd gpu_flow_build && \
    cmake \
        -D OpenCV_DIR=/usr/local/share/OpenCV \
        ../gpu_flow && \
    make -j $(nproc)

WORKDIR /video
ENTRYPOINT ["/src/gpu_flow_build/compute_flow"]
CMD ["--input-dir", "/video", "--ouput-dir", "/output"
