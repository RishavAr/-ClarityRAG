# Base image with Python
FROM python:3.10-slim

# Avoid Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies for FAISS and Transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]