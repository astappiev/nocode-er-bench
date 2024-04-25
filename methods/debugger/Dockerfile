FROM python:3

# Install worker script
WORKDIR /worker
COPY worker.py .

RUN pip install --no-cache-dir bullmq
CMD ["python", "-u", "./worker.py"]
