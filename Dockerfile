FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/compressed_video.py"]
CMD ["data/videos"]
