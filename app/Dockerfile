FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer. Editing main.py
# will not trigger a reinstall of TensorFlow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit reads config and credentials from ~/.streamlit
RUN mkdir -p /root/.streamlit
COPY config.toml /root/.streamlit/config.toml
COPY credentials.toml /root/.streamlit/credentials.toml

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
