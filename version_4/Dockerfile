FROM python:3.8-slim-bullseye AS compile-image

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m transformers.onnx --model=distilbert-base-cased onnx/distilbert-base-cased/

FROM python:3.8-slim-bullseye AS build-image
COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image onnx/distilbert-base-cased/model.onnx onnx/distilbert-base-cased/model.onnx

ENV PATH="/opt/venv/bin:$PATH"

COPY mytokenizer/ mytokenizer/
COPY version_4.py version_4.py
EXPOSE 8000
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 1000"

ENTRYPOINT exec gunicorn version_4:app --workers 1