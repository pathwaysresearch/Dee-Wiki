FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime assets — baked in from git working tree
COPY api/     ./api/
COPY data/    ./data/
COPY models/  ./models/
COPY Vault/   ./Vault/

ENV PORT=8080
EXPOSE 8080

CMD ["python", "api/index2.py", "--host", "0.0.0.0", "--port", "8080"]
