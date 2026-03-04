FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    python3-opencv python3-numpy \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY dcp_project.py /app/dcp_project.py

ENTRYPOINT ["python3", "/app/dcp_project.py"]