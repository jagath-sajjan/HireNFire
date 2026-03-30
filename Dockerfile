FROM python:3.11-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

# HF Spaces expects port 7860
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
