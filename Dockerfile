FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime@sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067

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

# Keep the base image's PyTorch/CUDA combination and install project dependencies.
# OLMo-core 1.6.3 requires NumPy < 2.0.
# Replace the base image's Ninja 1.11.1.1, whose wheel tags fail pip check.
RUN python -m pip install --no-cache-dir \
      'torch==2.5.1' \
      'numpy==1.26.4' \
      'ninja==1.11.1.4' \
      'ai2-olmo-core==1.6.3' \
      'PyYAML==6.0.3' \
      'transformers==5.18.0' \
      'tokenizers==0.23.1' \
      'wandb==0.30.0' \
    && python -m pip check

# Preserve the package layout used by comp.scicomp26 imports.
WORKDIR /workspace/comp/scicomp26
COPY . /workspace/comp/scicomp26/

# These import and version checks run during the build without requiring a GPU.
RUN python -c "import torch, numpy; from importlib.metadata import version; assert torch.__version__.split('+')[0] == '2.5.1'; assert torch.version.cuda == '12.1'; assert numpy.__version__ == '1.26.4'; assert version('ai2-olmo-core') == '1.6.3'" \
    && python scripts/train.py --help \
    && python scripts/run_200m_olmo_core.py --help

RUN mkdir -p /opt/scicomp26-manifest \
    && python -m pip freeze > /opt/scicomp26-manifest/requirements.actual.txt \
    && python --version > /opt/scicomp26-manifest/python-version.txt

CMD ["bash"]
