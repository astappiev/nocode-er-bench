FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "./entrypoint.py"]
CMD []
