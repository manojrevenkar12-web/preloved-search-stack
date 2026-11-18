# The search service. Models and the listings table are mounted (or synced from S3) at /srv/preloved.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir . && useradd -m search
USER search
ENV PRELOVED_HOME=/srv/preloved PRELOVED_WORLD=dev
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "preloved.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
