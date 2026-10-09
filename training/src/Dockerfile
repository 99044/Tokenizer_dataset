FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/tmp/huggingface \
    OMP_NUM_THREADS=1

# Read the compressed JSONL corpus during CPU preprocessing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/*

# Install the dependency closure exported from uv.lock. Keep the base image's
# CUDA 12.4 PyTorch wheel; replace its incompatible Ninja wheel.
COPY requirements.txt /opt/scicomp26-manifest/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/scicomp26-manifest/requirements.txt \
    && python -m pip check

# Apply the compatibility patch in a disposable build stage. The final image
# receives only the modified installed library, without repository source.
FROM runtime AS compatibility
COPY training/src/patch_olmo_core_torch26.py /tmp/patch_olmo_core_torch26.py
RUN python /tmp/patch_olmo_core_torch26.py

FROM runtime AS submission
COPY --from=compatibility /opt/conda/lib/python3.11/site-packages/olmo_core/nn/attention/backend.py /opt/conda/lib/python3.11/site-packages/olmo_core/nn/attention/backend.py
WORKDIR /workspace/comp/scicomp26
RUN python -c "import torch, numpy; from importlib.metadata import version; from olmo_core.nn.transformer import TransformerConfig; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'; assert numpy.__version__ == '1.26.4'; assert version('ai2-olmo-core') == '2.6.0'; assert hasattr(TransformerConfig, 'olmo3_1B')" \
    && python -m pip check
RUN mkdir -p /opt/scicomp26-manifest \
    && python -m pip freeze > /opt/scicomp26-manifest/requirements.actual.txt \
    && python --version > /opt/scicomp26-manifest/python-version.txt
CMD ["bash"]
