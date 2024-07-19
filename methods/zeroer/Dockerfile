FROM continuumio/miniconda3

RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY environment.yml .
RUN conda env create -f environment.yml

# this should be after pip install, to cache layers more efficient
COPY *.py .

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ZeroER", "python3", "-u", "entrypoint.py"]
CMD []
