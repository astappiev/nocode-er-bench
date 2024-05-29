FROM python:3.11

WORKDIR /workspace
COPY requirements.txt .
COPY *.py .

RUN pip install --no-cache-dir -r requirements.txt

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "./entrypoint.py"]
CMD []
