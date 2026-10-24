FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential python3 pkg-config && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
WORKDIR /workspace
# NVIDIA Container Toolkit e driver são responsabilidade do host.
CMD ["bash", "deploy/cuda/validate.sh"]
