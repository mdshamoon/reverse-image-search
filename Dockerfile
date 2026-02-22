FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV IMAGE_STORAGE_DIR=/app/data/images \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=items

RUN mkdir -p /app/data/images

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
