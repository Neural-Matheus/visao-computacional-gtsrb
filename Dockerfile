FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN chmod 1777 /tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        -r /tmp/requirements.txt \
        matplotlib \
        tqdm \
        ipywidgets \
        jupyterlab \
        notebook

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["bash", "-lc", "jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token=${JUPYTER_TOKEN:-changeme} \
    --ServerApp.root_dir=/workspace"]
