FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN conda install transformers  pandas tensorboardX scikit-learn --yes
RUN pip install pytorch_transformers
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/gpapadis/DLMatchers.git

# Install worker script
WORKDIR /workspace
COPY worker.py .
RUN pip install --no-cache-dir bullmq
CMD ["python3", "./worker.py"]
