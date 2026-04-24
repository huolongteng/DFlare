FROM tensorflow/tensorflow:2.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/DFlare

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.tflite.txt /tmp/requirements.tflite.txt

RUN python -m pip install --upgrade "pip==21.2.4" "setuptools<60" wheel && \
    python -m pip install -r /tmp/requirements.tflite.txt

COPY docker/entrypoint.sh /usr/local/bin/dflare-entrypoint
RUN chmod +x /usr/local/bin/dflare-entrypoint

COPY . .

RUN mkdir -p /workspace/DFlare/results

ENTRYPOINT ["/usr/local/bin/dflare-entrypoint"]
