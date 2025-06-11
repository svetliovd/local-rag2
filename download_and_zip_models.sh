#!/bin/bash

set -e

echo "Проверка за git-lfs..."
if ! command -v git-lfs &> /dev/null
then
    echo "🔧 Инсталираме git-lfs..."
    sudo apt update && sudo apt install git-lfs -y
    git lfs install
else
    echo "git-lfs вече е инсталиран."
fi

# Модели за теглене
MODELS=(
    "https://huggingface.co/BAAI/bge-m3"
    "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

for MODEL_URL in "${MODELS[@]}"
do
    MODEL_NAME=$(basename "$MODEL_URL")
    
    echo "Клониране на $MODEL_NAME..."
    git clone "$MODEL_URL"

    echo "Архивиране на $MODEL_NAME в ${MODEL_NAME}.zip..."
    zip -r "${MODEL_NAME}.zip" "$MODEL_NAME"

    echo "Готово: ${MODEL_NAME}.zip"
done

echo "Всички модели са изтеглени и архивирани успешно."
