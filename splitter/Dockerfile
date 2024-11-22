FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY DeepBlocker .
COPY splitter.py .

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENV PYTHONPATH="${PYTHONPATH}:/workspace/DeepBlocker"
ENTRYPOINT ["python", "-u", "splitter.py"]
CMD []
