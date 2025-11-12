# Multi-stage build за Football AI Service
FROM python:3.9-slim as base

# Метаданни
LABEL maintainer="Football AI Team"
LABEL version="1.0.0"
LABEL description="AI-powered football match predictions"

# Системни зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Работна директория
WORKDIR /app

# Копиране на requirements първо за по-добро кеширане
COPY requirements.txt .

# Инсталация на Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копиране на кода
COPY . .

# Създаване на необходими директории
RUN mkdir -p logs models data

# Експозиране на порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Стартиране на приложението
CMD ["python", "api/main.py"]

# Production stage с Gunicorn
FROM base as production

# Инсталация на Gunicorn
RUN pip install --no-cache-dir gunicorn

# Production команда
CMD ["gunicorn", "api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
