###############################################################################
# builder
###############################################################################
FROM python:3.12.7-slim-bookworm AS builder

RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /venv

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

FROM python:3.12.7-slim-bookworm AS mlflow

# ENV variables
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_ARTIFACT_S3_URI=$AWS_ARTIFACT_S3_URI
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /mlflow

# VIRTUAL_ENV = virtual environment path
# PATH = virtual environment executable path
ENV VIRTUAL_ENV=/venv/.venv \
    PATH="/venv/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Update and install needed packages
RUN apt-get update
RUN apt-get install -y curl unzip git

# Install AWS CLI for remote use
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscli.zip"
RUN unzip awscli.zip
RUN ./aws/install
RUN rm awscli.zip
RUN rm -rf ./aws

CMD mlflow server --port $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $AWS_ARTIFACT_S3_URI


###############################################################################
# dashboard
###############################################################################
FROM python:3.12.7-slim-bookworm AS dashboard

WORKDIR /app

ENV VIRTUAL_ENV=/venv/.venv \
    PATH="/venv/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN apt-get update && apt-get install -y curl & apt-get clean

COPY ./dashboard /app
COPY ./data/production/database_text_api.csv /app/data/production/database_text_api.csv

CMD streamlit run streamlit_app.py --server.port $PORT --server.address=0.0.0.0

###############################################################################
# api
###############################################################################
FROM python:3.12.7-slim-bookworm AS api

ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

ENV VIRTUAL_ENV=/venv/.venv \
    PATH="/venv/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN apt-get update & apt-get clean

COPY ./api /app
COPY ./data/production/database_text_api.csv /app/data/production/database_text_api.csv

CMD uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 90

###############################################################################
# python-cli
###############################################################################
FROM python:3.12.7-slim-bookworm AS python-cli

WORKDIR /app

RUN apt-get update && apt-get install -y make

# Install poetry
RUN pip install poetry==1.8.4

ENV VIRTUAL_ENV=/venv/.venv \
    PATH="/venv/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY Makefile /app

CMD ["bash"]

###############################################################################
# python-cli-3.8
###############################################################################
FROM python:3.8.20-slim-bullseye AS python-cli-3.8

WORKDIR /app

RUN apt-get update && apt-get install -y make && apt install -y libusb-1.0-0

COPY ./docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Makefile /app

CMD ["bash"]
