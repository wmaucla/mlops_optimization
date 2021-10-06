FROM python:3.8-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m transformers.onnx --model=distilbert-base-cased onnx/distilbert-base-cased/

COPY version_3.py version_3.py
EXPOSE 8000
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 1000"

ENTRYPOINT exec gunicorn version_3:app --workers 1