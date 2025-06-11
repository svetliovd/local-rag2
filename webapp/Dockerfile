FROM python:3.11-slim

# Системни зависимости
RUN apt-get update && apt-get install -y \
    git curl unzip build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Указване на работна директория
WORKDIR /app

# Копиране на файлове
COPY requirements.txt .
COPY app.py .
COPY entrypoint.sh .
COPY models/ ./models/
COPY chroma_db/ ./chroma_db/

# Инсталиране на Python зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod -R 777 /app/chroma_db

# Ollama моделът се предполага, че е предварително свален на хоста (в ~/.ollama)
# и Ollama сървърът работи локално и достъпен в контейнера

# Старт
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

