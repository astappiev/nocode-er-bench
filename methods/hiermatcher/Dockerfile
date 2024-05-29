FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y unzip build-essential wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir deepmatcher

WORKDIR /workspace

# Download the EntityMatcher code
RUN wget https://github.com/nishadi/EntityMatcher/archive/refs/heads/master.zip && \
    unzip master.zip && \
    mv EntityMatcher-master/* . && \
    rm -rf master.zip EntityMatcher-master && \
    rm -rf data # Remove datasets

# Download the pre-trained model on English Wikipedia
ADD https://zenodo.org/record/6466387/files/wiki.en.bin /workspace/embedding

COPY requirements.txt .
COPY *.py .

RUN pip install --no-cache-dir -r requirements.txt

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "./entrypoint.py"]
CMD []
