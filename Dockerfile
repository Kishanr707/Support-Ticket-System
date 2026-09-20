# Dockerfile — packages the FastAPI backend (ML model + /predict API)
#
# Only src/ and the trained model artifact are copied in. Raw training
# data (data/) is NOT included — it's not needed to serve predictions,
# only to retrain the model, and skipping it keeps the image much
# smaller (data/issues.csv alone is ~20MB).

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this
# step and skip reinstalling every time only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the already-trained model artifact
COPY src/ src/
COPY models/ models/

EXPOSE 8000

# --host 0.0.0.0 is required here, not 127.0.0.1 — inside a container,
# binding to 127.0.0.1 makes the server unreachable from outside the
# container even with a port mapping. This is the #1 Docker networking
# mistake people make on their first container.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]