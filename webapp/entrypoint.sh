#!/bin/bash
echo "Стартиране на Streamlit офлайн..."

# Стартиране на приложението
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
