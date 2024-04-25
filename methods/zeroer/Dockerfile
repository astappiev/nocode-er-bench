FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y git build-essential
RUN git clone https://github.com/nishadi/zeroer.git
WORKDIR zeroer
RUN conda env create -f environment.yml
