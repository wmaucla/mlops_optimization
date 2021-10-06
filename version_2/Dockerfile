FROM python:3.8-slim-bullseye

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY version_2.py version_2.py


EXPOSE 8000
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 1000"

ENTRYPOINT exec gunicorn version_2:app --workers 1