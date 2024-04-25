FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git build-essential
RUN conda install -c conda-forge nvidia-apex
RUN git clone https://github.com/nishadi/ditto.git
WORKDIR ditto
RUN sed -i '8d' requirements.txt
RUN pip install -r requirements.txt
RUN pip install typer --upgrade
RUN python -m spacy download en_core_web_lg
RUN pip install tensorboardX nltk
RUN python -m nltk.downloader stopwords
WORKDIR /home/remote/u6852937/projects
