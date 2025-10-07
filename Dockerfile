FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN adduser --disabled-password --gecos '' appuser

RUN mkdir -p /data \
 && chown -R appuser:appuser /app /data

# Копируем код с правильным владельцем
COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "bot.py"]