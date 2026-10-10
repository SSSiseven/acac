# syntax=docker/dockerfile:1
# Official Python 3.9.25 image; verified multi-platform digest.
FROM python:3.9.25-slim-bookworm@sha256:a02e9c5406c416c504d6c9a1a306ff4080c3173f1008d192f953bd20382a2d5c

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ACAC_DATA_DIR=/data \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# pygame is eagerly imported by Overcooked; no desktop is required.
# CUDA libraries come with PyTorch; NVIDIA drivers stay on the host.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libgl1 libglib2.0-0 \
        libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-container.txt ./
RUN python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install -r requirements-container.txt \
    && python -m pip check

# Run from source: setup.py contains unrelated legacy dependencies.
COPY . ./
# Build checks need no GPU and do not start training.
RUN python -c "import torch; assert torch.__version__ == '2.1.0+cu121'; assert torch.version.cuda == '12.1'" \
    && python -m unittest discover -s tests -q \
    && python acac/scripts/acac_main.py --help > /dev/null \
    && mkdir -p /data /usr/local/share/tg-acac \
    && python -m pip freeze > /usr/local/share/tg-acac/requirements-resolved.txt

# Override CMD with an explicit training or analysis command.
CMD ["python", "acac/scripts/acac_main.py", "--help"]
