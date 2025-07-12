FROM mcr.microsoft.com/devcontainers/python:1-3.11

RUN apt-get update && apt-get install -y git

WORKDIR /workspaces/${localWorkspaceFolderBasename}

COPY requirements.txt ./
RUN pip install -r requirements.txt

