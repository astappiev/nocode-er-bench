# Can we use pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime instead? it should be 3 times smaller
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

WORKDIR /methods
RUN git clone https://github.com/gpapadis/DLMatchers.git
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.8.0 -c pytorch -c conda-forge
RUN pip install pytorch_transformers scikit-learn tensorboardX pandas transformers

# Install worker script
WORKDIR /worker
COPY worker.py .
RUN pip install --no-cache-dir bullmq
CMD ["python3", "./worker.py"]
