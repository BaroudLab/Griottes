# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN pip install .
RUN python -m pytest -v