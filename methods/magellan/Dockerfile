FROM python:3.11

RUN pip install --no-cache-dir py-entitymatching

WORKDIR /worker
COPY parser.py .
COPY transform.py .
COPY entrypoint.py .

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "./entrypoint.py"]
CMD []
