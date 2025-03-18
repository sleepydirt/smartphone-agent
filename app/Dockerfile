FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update -y && \
    apt-get install -y libpq-dev gcc && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ./smartphone_inventory_sgd.csv /tmp/smartphone_inventory_sgd.csv

# gradio port
EXPOSE 7860

# FastAPI port
EXPOSE 8000