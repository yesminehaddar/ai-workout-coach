FROM python:3.11-slim

# Bibliothèques système nécessaires à OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY backend/ backend/
RUN pip install --no-cache-dir .

COPY frontend/ frontend/
COPY .streamlit/ .streamlit/
# Le code est lu depuis /app/backend : les chemins data/ se résolvent depuis /app.
ENV PYTHONPATH=/app/backend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
