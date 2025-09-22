FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV PYTHONPATH="/app/src"

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD [\
    "python",\
    "-c",\
    "import sys, urllib.request; resp = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4); sys.exit(0 if resp.status == 200 else 1)"\
]

CMD [
    "uvicorn",
    "highest_volatility.app.api:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]
