# ---- Base image ----
FROM python:3.12-slim

# ---- System dependencies ----
# build-essential: needed to compile some Python packages (e.g. numpy/torch deps)
# libgl1: required by some OCR/image libs (opencv-related deps pulled in by easyocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Install Python dependencies first (better layer caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy the rest of the project (includes src/, data/, template/, static/) ----
COPY . .

# ---- Environment ----
ENV PYTHONUNBUFFERED=1
# GROQ_API_KEY should be passed at runtime, not baked into the image (see below)

# ---- Expose the port uvicorn will run on ----
EXPOSE 8000

# ---- Start the server ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]