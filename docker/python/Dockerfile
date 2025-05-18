# docker/python/Dockerfile
FROM python:3.12-slim
ENV TZ=UTC PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential git
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /work
