FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /worker
COPY . .

RUN pip install --no-cache-dir -r DeepBlocker/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "./entrypoint.py"]
CMD []
