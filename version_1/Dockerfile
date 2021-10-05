FROM python:3.8.9

WORKDIR /app
COPY requirements.txt requirements.txt
COPY version_1.py version_1.py

RUN pip install -r requirements.txt

EXPOSE 8000
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 1000"

ENTRYPOINT exec gunicorn version_1:app --workers 1