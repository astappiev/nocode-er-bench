FROM python:3

# Install worker script
WORKDIR /workspace
COPY worker.py .

RUN pip install --no-cache-dir bullmq
CMD ["python", "-u", "./worker.py"]
