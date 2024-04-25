FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y unzip build-essential wget git
RUN conda create -n deepmatcher python=3.6 -y
SHELL ["conda", "run", "-n", "deepmatcher", "/bin/bash", "-c"]
RUN pip install deepmatcher
RUN mkdir /root/.vector_cache
RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip --directory-prefix=/root/.vector_cache
RUN unzip /root/.vector_cache/wiki.en.zip -d /root/.vector_cache/
RUN rm /root/.vector_cache/wiki.en.vec
RUN rm /root/.vector_cache/wiki.en.zip
WORKDIR /methods
RUN git clone https://github.com/nishadi/EntityMatcher.git
WORKDIR /methods/EntityMatcher/embedding/
RUN wget https://zenodo.org/record/6466387/files/wiki.en.bin
