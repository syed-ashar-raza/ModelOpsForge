FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN python -m modelopsforge.cli train

EXPOSE 8000

CMD ["uvicorn", "modelopsforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
