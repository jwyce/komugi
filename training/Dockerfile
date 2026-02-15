# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── Build Rust on the SAME base image to avoid glibc/OpenSSL mismatches ──
FROM --platform=linux/amd64 ${BASE_IMAGE} AS rust-builder

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config clang libclang-dev curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.88.0
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/komugi-core crates/komugi-core
COPY crates/komugi-engine crates/komugi-engine

# Stub out komugi-wasm so workspace resolves
RUN mkdir -p crates/komugi-wasm/src && \
    printf '' > crates/komugi-wasm/src/lib.rs && \
    printf '[package]\nname = "komugi-wasm"\nversion = "0.1.0"\nedition = "2021"\n' > crates/komugi-wasm/Cargo.toml

RUN cargo build --release --bin selfplay --features neural -p komugi-engine
RUN mkdir -p /out && install -m 0755 target/release/selfplay /out/selfplay

# ── Runtime ──
FROM --platform=linux/amd64 ${BASE_IMAGE} AS runtime

WORKDIR /workspace

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN ORT_DIR="$(python -c "\
import onnxruntime, pathlib; \
print(pathlib.Path(onnxruntime.__file__).resolve().parent/'capi')")" && \
    NVIDIA_DIR="$(python -c "import nvidia; print(nvidia.__path__[0])")" && \
    mkdir -p /usr/local/lib && \
    ln -sf "$ORT_DIR"/libonnxruntime*.so* /usr/local/lib/ && \
    find "$NVIDIA_DIR" -name 'lib' -type d | tee /etc/ld.so.conf.d/nvidia-pip.conf && \
    echo /usr/local/lib >> /etc/ld.so.conf.d/nvidia-pip.conf && \
    echo /opt/conda/lib >> /etc/ld.so.conf.d/nvidia-pip.conf && \
    ldconfig && \
    ldd /usr/local/lib/libonnxruntime_providers_cuda.so | grep -c "not found" | grep -q "^0$"

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    PYTHONUNBUFFERED=1

COPY training/model.py .
COPY training/train.py .
COPY training/export_onnx.py .
COPY training/preprocess.py .
COPY training/full_loop.sh .
RUN chmod +x full_loop.sh

COPY --from=rust-builder /out/selfplay /usr/local/bin/selfplay

CMD ["/bin/bash"]
