# DGX Spark / GB10 (Grace Blackwell) - ARM64 + CUDA 13
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    git python3 python3-pip wget make g++ cmake ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# libcuda.so Stub für Build-Zeit
RUN ln -sf ${CUDA_HOME}/lib64/stubs/libcuda.so ${CUDA_HOME}/lib64/stubs/libcuda.so.1 \
    && echo "${CUDA_HOME}/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

# Wyoming Python-Pakete
RUN pip3 install --no-cache-dir --break-system-packages wyoming

# whisper.cpp bauen
RUN git clone https://github.com/ggerganov/whisper.cpp.git /app/whisper-core

WORKDIR /app/whisper-core
RUN cmake -B build \
    -DGGML_CUDA=1 \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_HOME}/lib64/stubs -lcuda" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_HOME}/lib64/stubs -lcuda" \
    && cmake --build build --config Release -j$(nproc)

# Server-Script aus dem Repo laden
COPY server.py /app/server.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/server.py"]
