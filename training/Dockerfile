FROM --platform=linux/amd64 rust:1.88-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/komugi-core crates/komugi-core
COPY crates/komugi-engine crates/komugi-engine

RUN mkdir -p crates/komugi-wasm/src && \
    echo "" > crates/komugi-wasm/src/lib.rs && \
    printf '[package]\nname = "komugi-wasm"\nversion = "0.1.0"\nedition = "2021"\n' > crates/komugi-wasm/Cargo.toml

RUN cargo build --release --bin selfplay --features neural -p komugi-engine && \
    cp target/release/selfplay /usr/local/bin/selfplay

FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/model.py .
COPY training/train.py .
COPY training/export_onnx.py .
COPY training/full_loop.sh .
RUN chmod +x full_loop.sh

COPY --from=builder /usr/local/bin/selfplay /usr/local/bin/selfplay

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
