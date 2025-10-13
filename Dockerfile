# Use lightweight Python base image
FROM python:3.12-slim

# ------------------------------------------------------------------
# Install essential build tools and dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    libblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Set environment variables
# ------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# ------------------------------------------------------------------
# Copy project files
# ------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------
# Run the Django app using Gunicorn (Cloud Run listens on $PORT)
# ------------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--preload", "pathpilot.wsgi:application"]
