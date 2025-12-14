FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \ 
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 libxext6 libgl1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /workspace

CMD ["python", "code/test_CNN.py", "--dataset", "DRIVE", "--skip_restore", "--limit_images", "2"]
