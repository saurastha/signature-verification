FROM python:3.11-slim

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1

WORKDIR /sign-verification

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && python3.11 -m venv ${VIRTUAL_ENV} \
    && pip install --upgrade pip setuptools \
    && poetry install --without dev \
    && rm -rf ${HOME}/.cache/*

COPY app ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]