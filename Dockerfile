FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[serve,gpu]"

EXPOSE 8000
VOLUME ["/exports"]

ENTRYPOINT ["tether"]
CMD ["serve", "/exports", "--host", "0.0.0.0", "--port", "8000"]
