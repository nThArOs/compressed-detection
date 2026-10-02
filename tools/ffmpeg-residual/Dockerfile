# FFmpeg 7.1 decoding H.264 only, patched to dump the luma residual of each frame (see patch.py).
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential nasm python3 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone -q --depth 1 --branch n7.1.1 https://github.com/FFmpeg/FFmpeg.git /ff
COPY patch.py /patch.py
RUN python3 /patch.py /ff
WORKDIR /ff
RUN ./configure --disable-autodetect --disable-network --disable-doc --disable-ffplay --disable-ffprobe --enable-ffmpeg \
 && make -j"$(nproc)" && cp ffmpeg /usr/local/bin/ffmpeg-residual
ENTRYPOINT ["ffmpeg-residual"]
